import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from monai.data import DataLoader, decollate_batch, ImageDataset
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    Resize,
    ToTensor,
    ScaleIntensityRange,
)
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.utils import set_determinism
from torch.utils.data import random_split

from monai.transforms import Resize
import nibabel as nib

import torch

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
from time import time

############
# This code is using torch.multiprocessing to work
# execute this code with:
# python monai-ddp.py
############

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class SegmentationModel(torch.nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.model = model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.threshold = threshold

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        """
        Predicts the segmentation of the input image.

        Args:
            x (torch.Tensor): The input image.
        """
        x = self.forward(x)
        x = torch.sigmoid(x)
        # Threshold output to 0 or 1
        x = (x > self.threshold).float()
        return x

def get_datasets(directory):
    """
    Get the training, validation, and test datasets.
    """
    # For preprocessed images
    train_transforms = None
    train_transforms_seg = None

    ct_scans_directory = os.path.join(directory, "volume")
    segs_directory = os.path.join(directory, "seg")
    ct_scans = os.listdir(ct_scans_directory)
    segs = os.listdir(segs_directory)
    ct_scans.sort()
    segs.sort()

    num_images = 10#len(ct_scans)

    # Create a training data loader
    dataset = ImageDataset(
        image_files=[os.path.join(ct_scans_directory, ct_scans[i]) for i in np.arange(num_images)],
        seg_files=[os.path.join(segs_directory, segs[i]) for i in np.arange(num_images)],
        transform=train_transforms,
        seg_transform=train_transforms_seg,
        reader="NibabelReader"
    )

    # Train test split
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [0.7, 0.2, 0.1]
    )

    batch_size = 1

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size= batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size= batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=val_sampler,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size= batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(test_dataset),
    )
    return train_loader, val_loader, test_loader

def main(rank, world_size, max_epochs=20):
    setup(rank, world_size)
    train_loader, val_loader, test_loader = get_datasets("./data/train-512/preprocessed")

    model = SegmentationModel().to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = DiceCELoss(sigmoid=True)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )
    metric = DiceMetric(include_background=True, reduction="mean")
    print("Starting training")

    directory = ""

        # Training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    for epoch in range(max_epochs):
        train_loader.sampler.set_epoch(epoch)
        epoch_loss = 0
        for batch_data in train_loader:
            model.train()
            inputs, segs = (
                batch_data[0].to(rank, non_blocking=True),
                batch_data[1].to(rank, non_blocking=True),
            )
            optimizer.zero_grad()
            # with torch.cuda.amp.autocast():
            outputs = model.forward(inputs)
            loss = loss_function(outputs, segs)
            loss.backward()
            optimizer.step()
            ## Optimization with automatic mixed precision
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            epoch_loss += loss.item()
            print(f"{epoch + 1}/{max_epochs}, batch_train_loss: {loss.item():.4f}")

        epoch_loss = torch.tensor([epoch_loss]).to(rank)
        dist.reduce(epoch_loss, 0, op=dist.ReduceOp.SUM)
        # Assuming there is no padding
        epoch_loss_values.append((epoch_loss / len(train_loader)*world_size).item())
        if rank == 0:
            print(
                f"{epoch + 1}/{max_epochs}, train_loss: {epoch_loss_values[-1]:.4f}"
            )

        ## Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                value_total = 0
                for val_data in val_loader:
                    val_inputs, val_segs = (
                        val_data[0].to(rank, non_blocking=True),
                        val_data[1].to(rank, non_blocking=True),
                    )
                    val_outputs = model.module.predict(val_inputs)
                    value = metric(val_outputs, val_segs)
                    value_total += value.item()
                    print(f"{epoch + 1}/{max_epochs}, validation_dice: {value.item():.4f}")
                metric_values.append(value_total / len(val_loader))
                # Save best metric model
                # Also if on the same GPU (rank 0)
                if value_total > best_metric and rank == 0:
                    best_metric = value_total
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.module.state_dict(),
                        os.path.join(directory, "model/best_metric_model.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"{epoch + 1}/{max_epochs}, mean validation_dice: {value_total/len(val_loader):.4f}"
                    )
            # Scheduler step
            lr_scheduler.step(metric_values[-1])

    if rank == 0:
        print(
            f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
        )
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    deb = time()
    mp.spawn(main, args=(world_size,1), nprocs=world_size)
    print("Time taken: ", time()-deb)