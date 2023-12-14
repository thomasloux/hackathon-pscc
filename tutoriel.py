import os
import json
import shutil
import tempfile
import time
from typing import Tuple
import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.networks.layers import Norm
from monai.transforms import (
    AsDiscrete,
    Activations,
    ScaleIntensityRanged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRange,
    RandCropByPosNegLabeld,
    Compose,
    SpatialCropd,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR, UNet

from monai import data
from monai.data import decollate_batch, DataLoader, Dataset
from functools import partial

import torch
import torch.optim as optim
import torch.nn as nn

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from sklearn.model_selection import train_test_split

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def get_loader(batch_size, data_dir, roi):
    """
    Provide data loader for training and validation data.
    """
    images = sorted(glob.glob(os.path.join(data_dir, "volume", "*.nii.gz")))
    segs = sorted(glob.glob(os.path.join(data_dir, "seg", "*.nii.gz")))

    data_dicts = [
        {"image": image_name, "label": seg_name}
        for image_name, seg_name in zip(images, segs)
    ]
    # data_dicts = data_dicts[:20] # Limit data to test the pipeline

    train_files, val_files = train_test_split(data_dicts, train_size=0.8, random_state=0)

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            SpatialCropd(keys=["image", "label"], roi_start=(30, 30, 0), roi_end=(512-30, 512-100, 130)),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image"
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            # Probably not needed 
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1024,
                a_max=3071,
                b_min=0.0,
                b_max=1.0,
                clip=True),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            SpatialCropd(keys=["image", "label"], roi_start=(30, 30, 0), roi_end=(512-30, 512-100, 130)),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image"
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            # Probably not needed 
            ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0.0, b_max=1.0, clip=True),
        ]
    )

    train_ds = data.CacheDataset(data=train_files, transform=train_transform, cache_rate=0.4, num_workers=8)
    val_ds = data.CacheDataset(data=val_files, transform=val_transform, cache_rate=0.4, num_workers=8)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=DistributedSampler(train_ds, shuffle=True),
    )

    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=DistributedSampler(val_ds, shuffle=False),
    )

    return train_loader, val_loader

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        loss_function: torch.nn.Module,
        metric: torch.nn.Module,
        gpu_id: int,
        save_every: int,
        folder_save: str,
        roi: Tuple[int, int, int],
        val_interval: int = 2,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.metric = metric
        self.save_every = save_every
        self.folder_save = folder_save
        self.roi = roi
        self.model = DDP(model, device_ids=[gpu_id])
        self.val_interval = val_interval

        self.mean_loss_history = []
        self.mean_metric_history = []

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_function(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))["image"])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        total_loss = 0
        for batch_data in tqdm(self.train_data):
            inputs, labels = (
                batch_data["image"].to(self.gpu_id),
                batch_data["label"].to(self.gpu_id),
            )
            total_loss += self._run_batch(inputs, labels)
        total_loss /= len(self.train_data)
        self.mean_loss_history.append(total_loss)
        print(f"Epoch {epoch} | Training loss: {total_loss}")

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = os.path.join(self.folder_save, "checkpoint.pt")
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def _validate(self, epoch):
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([AsDiscrete(to_onehot=2)])
        self.val_data.sampler.set_epoch(epoch)
        self.model.eval()
        with torch.no_grad():
            for batch_data in self.val_data:
                inputs, labels = (
                    batch_data["image"].to(self.gpu_id),
                    batch_data["label"].to(self.gpu_id),
                )
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    inputs, self.roi, sw_batch_size, self.model, overlap=0.7, sw_device=self.gpu_id, device=self.gpu_id, mode="gaussian"
                )
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(labels)]
                
                self.metric(y_pred=val_outputs, y=val_labels)
            
            # Aggregate the final mean metric value
            value_metric = self.metric.aggregate().item()
            print(f"Epoch {epoch} | Validation metric: {value_metric}")
            self.mean_metric_history.append(value_metric)
            self.metric.reset()
            return value_metric

        self.model.train()

    def _plot(self):
        """
        Saves the plot of the loss and metric values for each epoch.
        """
        plt.figure("train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(self.mean_loss_history))]
        y = self.mean_loss_history
        plt.xlabel("epoch")
        plt.plot(x, y, color="red")
        plt.subplot(1, 2, 2)
        plt.title("Dice Score")
        x = [i + 1 for i in range(len(self.mean_metric_history))]
        y = self.mean_metric_history
        plt.xlabel("epoch")
        plt.plot(x, y, color="green")
        plt.savefig(os.path.join(self.folder_save, "loss_metric.png"))

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            
            if epoch % self.val_interval == 0:
                value_metric = self._validate(epoch)

            self.lr_scheduler.step(value_metric)
        
        if self.gpu_id == 0:
            self._plot()

def main(
        rank: int,
        world_size: int,
        save_every: int,
        total_epochs: int,
        batch_size: int,
        data_dir: str,
        folder_save: str,):

    # Set up distributed training
    ddp_setup(rank, world_size)

    roi = (192, 192, 64)

    # Load data
    train_loader, val_loader = get_loader(batch_size, data_dir, roi)

    # Create model
    model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.2,
    ).to(rank)
    model = DDP(model, device_ids=[rank])

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, verbose=True
    )

    # Define loss function
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    metric = DiceMetric(include_background=False, reduction="mean")

    # Train
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        loss_function,
        metric,
        rank,
        save_every,
        folder_save,
        roi,
        val_interval=1
    )
    trainer.train(total_epochs)

    # Clean up distributed training
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total-epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save-every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch-size', default=2, type=int, help='Input batch size on each device (default: 2)')
    parser.add_argument('--data-dir', default="/tsi/data_education/data_challenge/train", type=str, help='Path to data directory')
    parser.add_argument('--folder-save', default="model/default", type=str, help='Path to save checkpoints')
    args = parser.parse_args()

    if not os.path.exists(args.folder_save):
        os.makedirs(args.folder_save)
    
    world_size = torch.cuda.device_count()
    arguments = (world_size, args.save_every, args.total_epochs, args.batch_size, args.data_dir, args.folder_save)
    mp.spawn(main, args=arguments, nprocs=world_size)
