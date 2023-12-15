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
    AsDiscrete
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
from torch.distributed.elastic.multiprocessing.errors import record

import logging
logging.basicConfig(level=logging.INFO)

############
# This code is using torchrun to work
# execute this code with:
# torchrun monai-ddp.py
############

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
    # For preprocessed images
    train_transforms = None
    train_transforms_seg = Compose([
        AsDiscrete(rounding="torchrounding")
    ])

    ct_scans_directory = os.path.join(directory, "volume")
    segs_directory = os.path.join(directory, "seg")
    ct_scans = os.listdir(ct_scans_directory)
    segs = os.listdir(segs_directory)
    ct_scans.sort()
    segs.sort()

    num_images = len(ct_scans)

    # Create a training data loader
    dataset = ImageDataset(
        image_files=[os.path.join(ct_scans_directory, ct_scans[i]) for i in np.arange(num_images)],
        seg_files=[os.path.join(segs_directory, segs[i]) for i in np.arange(num_images)],
        transform=train_transforms,
        seg_transform=train_transforms_seg,
        reader="NibabelReader"
    )

    # Train test split
    train_dataset, val_dataset = random_split(
        dataset, [0.8, 0.2,]
    )

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

@record
def main(local_rank: int, world_size: int, folder_save: str, max_epochs=20):
    setup(local_rank, world_size)

    global_rank = local_rank # Eviter de tout réécrire
    # local_rank = int(os.environ["LOCAL_RANK"])
    # global_rank = int(os.environ["RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    logging.info(f"local_rank: {local_rank}, global_rank: {global_rank}, world_size: {world_size}")

    train_loader, val_loader = get_datasets("./data/train-512/preprocessed")

    model = SegmentationModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, verbose=True
    )
    metric = DiceMetric(include_background=False, reduction="mean")

    directory = ""

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
                batch_data[0].to(local_rank, non_blocking=True),
                batch_data[1].to(local_rank, non_blocking=True),
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
            # segs_onehot = [post_label(i) for i in decollate_batch(segs)]
            # outputs_softmax = [post_pred(i) for i in decollate_batch(outputs)]
            # metric_value_test = metric(y_pred=outputs_softmax, y=segs_onehot)
            # print(f"GPU {global_rank}: {epoch + 1}/{max_epochs}, batch_train_loss: {loss.item():.4f}, batch_train_dice: {torch.mean(metric_value_test).item():.4f}")
            epoch_loss += loss.item()
            # print(f"GPU {global_rank}: {epoch + 1}/{max_epochs}, batch_train_loss: {loss.item():.4f}")
        print(f"GPU {global_rank}: {epoch + 1}/{max_epochs}, train_loss: {epoch_loss/len(train_loader):.4f}")
        #epoch_loss = torch.tensor([epoch_loss]).to(local_rank)/len(train_loader)
        #dist.reduce(epoch_loss, 0, op=dist.ReduceOp.SUM)
        # Assuming there is no padding
        epoch_loss_values.append(epoch_loss/len(train_loader))
        #metric.reset()

        ## Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_segs = (
                        val_data[0].to(local_rank, non_blocking=True),
                        val_data[1].to(local_rank, non_blocking=True),
                    )
                    val_outputs = model(val_inputs)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_segs = [post_label(i) for i in decollate_batch(val_segs)]
                    metric(y_pred=val_outputs, y=val_segs)
                    # print(f"GPU {global_rank}: {epoch + 1}/{max_epochs}, validation_dice: {value.item():.4f}")
                # value_total = torch.tensor([value_total]).to(local_rank)/len(val_loader)
                #dist.reduce(value_total, 0, op=dist.ReduceOp.SUM)
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

    args = parser.parse_args()

    world_size = torch.cuda.device_count()

    directory = ""
    if not os.path.exists(os.path.join(directory, "model", args.folder_save)):
        print("Creating directory")
        os.mkdir(os.path.join(directory, "model", args.folder_save))

    #main(folder_save=args.folder_save, max_epochs=args.epochs)
    mp.spawn(main, args=(world_size, args.folder_save, args.epochs), nprocs=world_size)