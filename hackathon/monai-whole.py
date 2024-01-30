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
    AsDiscrete,
    EnsureChannelFirst

)
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.utils import set_determinism
from torch.utils.data import random_split

from monai.transforms import Resize
import nibabel as nib
from typing import List
import torch

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
from time import time

import argparse

from model import SegmentationModel

import logging
logging.basicConfig(level=logging.INFO)

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '20355'

    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def get_datasets(directory):
    """
    Get the training, validation, and test datasets.
    """


    ct_scans = sorted(glob.glob(os.path.join(directory, "volume", "*"))) # Capture .nii or .nii.gz files
    segs = sorted(glob.glob(os.path.join(directory, "seg", "*"))) # Capture .nii or .nii.gz files

    data_dicts = [{"image": image, "label": seg} for image, seg in zip(ct_scans, segs)]

    # To easily change the number of images used for training
    # Helpful for debugging using only a few images
    num_images = len(ct_scans)

    train_files, val_files = train_test_split(data_dicts, train_size=0.8, random_state=0)

    # For preprocessed images
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(a_min=-1024, a_max=3071, b_min=0.0, b_max=1.0, clip=True),
            SpatialCropd(roi_start=(30, 30, 0), roi_end=(512-30, 512-100, 130)),
            DivisiblePadd(pad_size=16, mode="constant"),

        ]
    )

    train_dataset = Dataset(data=train_files, transform=train_transforms)
    val_dataset = Dataset(data=val_files, transform=train_transforms)

    batch_size = 3

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size= batch_size,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size= 3,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader

def save_plot(epoch_loss_values: List[float], metric_values:List[float], folder_save: str):
    """
    Saves the plot of the loss and metric values for each epoch.
    """
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Dice Score")
    x = [i + 1 for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="green")
    plt.savefig(os.path.join(folder_save, "loss_metric.png"))

def main(local_rank: int, world_size: int, folder_save: str, data_dir, max_epochs=20):
    setup(local_rank, world_size)

    global_rank = local_rank # Eviter de tout réécrire

    logging.info(f"local_rank: {local_rank}, global_rank: {global_rank}, world_size: {world_size}")

    train_loader, val_loader = get_datasets(data_dir)

    # model = UNet(
    #         spatial_dims=3,
    #         in_channels=1,
    #         out_channels=2,
    #         channels=(16, 32, 64, 128, 256),
    #         strides=(2, 2, 2, 2),
    #         num_res_units=2,
    #         norm=Norm.BATCH,
    #         dropout=0.2,
    # ).to(rank)
    model = SwinUNETR(
        img_size=roi,
        in_channels=1,
        out_channels=2,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        drop_rate=0.3,
        use_v2=True,
    ).to(rank)
    model = DDP(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=100, verbose=True
    )
    metric = DiceMetric(include_background=False, reduction="mean")

    directory = "../"

    print("Starting training")

        # Training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []


    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    for epoch in range(max_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            model.train()
            inputs, segs = (
                batch_data["image"].to(local_rank, non_blocking=True),
                batch_data["label"].to(local_rank, non_blocking=True),
            )
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = loss_function(outputs, segs)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            # print(f"GPU {global_rank}: {epoch + 1}/{max_epochs}, batch_train_loss: {loss.item():.4f}")
        print(f"GPU {global_rank}: {epoch + 1}/{max_epochs}, train_loss: {epoch_loss/len(train_loader):.4f}")

        epoch_loss_values.append(epoch_loss/len(train_loader))

        ## Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_segs = (
                        val_data["image"].to(local_rank, non_blocking=True),
                        val_data["label"].to(local_rank, non_blocking=True),
                    )
                    val_outputs = model(val_inputs)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_segs = [post_label(i) for i in decollate_batch(val_segs)]
                    metric(y_pred=val_outputs, y=val_segs)
                    # print(f"GPU {global_rank}: {epoch + 1}/{max_epochs}, validation_dice: {value.item():.4f}")

                mean_value = metric.aggregate().item()
                metric_values.append(mean_value)
                metric.reset()
                print(f"GPU {global_rank}: {epoch + 1}/{max_epochs}, validation_dice: {mean_value:.9f}")

                # Save best metric model
                # Also if on the same GPU (rank 0)
                if mean_value > best_metric and local_rank == 0:
                    best_metric = mean_value
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.module.state_dict(),
                        os.path.join(directory, "model", folder_save, "best_metric_model.pth"),
                    )
                    print("saved new best metric model")
            # Scheduler step
            lr_scheduler.step(metric_values[-1])

    if local_rank == 0:
        print(
            f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
        )
        save_plot(epoch_loss_values, metric_values, os.path.join(directory, "model", folder_save))

    with open(os.path.join(directory, "model", folder_save, "train_loss.txt"), "w") as f:
        f.write("Training loss\n")
        for i, item in enumerate(epoch_loss_values):
            f.write(f"Epoch {i+1}: {item}\n")
        f.write("Validation dice\n")
        for i, item in enumerate(metric_values):
            f.write(f"Epoch {i+1}: {item}\n")

    dist.destroy_process_group()

if __name__ == "__main__":
    # Argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--folder-save", type=str)
    parser.add_argument("--data-dir", type=str)

    args = parser.parse_args()

    world_size = torch.cuda.device_count()

    # globale directory
    directory = "../"
    if not os.path.exists(os.path.join(directory, "model", args.folder_save)):
        print("Creating directory")
        os.mkdir(os.path.join(directory, "model", args.folder_save))

    mp.spawn(main, args=(world_size, args.folder_save, args.epochs, args.data_dir), nprocs=world_size)