import os
import json
import shutil
import tempfile
import time
from typing import Tuple
import glob

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
    RandRotate90d,
    SpatialCropd,
    KeepLargestConnectedComponent,
    Compose
)
from monai.networks.nets import UNet
from monai.data import Dataset, DataLoader, decollate_batch
from monai.config import print_config
from monai.metrics import DiceMetric

import torch
import torch.optim as optim
import torch.nn as nn

import nibabel as nib
import numpy as np
from multiprocessing import Pool

# Visualizations
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from tqdm import tqdm
sns.set_style("darkgrid")

def recursive_removal_module(input_dict):
    newdict = {}
    for key in input_dict.keys():
        newKey = key[7:]
        if isinstance(input_dict[key], dict):
            newdict[newKey] = recursive_removal_module(input_dict[key])
        else:
            newdict[newKey] = input_dict[key]
    return newdict

def get_loader(data_dir):
    """
    Provide data loader for training and validation data.
    """
    images = sorted(glob.glob(os.path.join(data_dir, "volume", "*.nii.gz")))
    segs = sorted(glob.glob(os.path.join(data_dir, "seg", "*.nii.gz")))

    data_dicts = [
        {"image": image_name, "label": seg_name}
        for image_name, seg_name in zip(images, segs)
    ]

    data_dicts = data_dicts

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
           # SpatialCropd(keys=["image", "label"], roi_start=(30, 30, 0), roi_end=(512-30, 512-100, 130)),
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

    dataset = Dataset(data=data_dicts, transform=val_transform)

    val_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=6,
    )

    return val_loader

def show_irm(data: np.ndarray, inputs, val_outputs, name=None):
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Initialize images for each subplot
    im1 = axs[0].imshow(data[:, :, 0], cmap='gray', vmin=0, vmax=1)
    im2 = axs[1].imshow(inputs[:, :, 0], vmin=0, vmax=1)
    im3 = axs[2].imshow(val_outputs[:, :, 0], vmin=0, vmax=1)

    # Function to update the plots for each frame
    def update(frame):
        im1.set_data(data[:, :, frame])
        axs[0].set_title(f'CT scans {frame}')
        axs[0].grid(False)

        im2.set_data(inputs[:, :, frame])
        axs[1].set_title(f"Real Mask (cancer cells) {frame}")
        axs[1].grid(False)

        im3.set_data(val_outputs[:, :, frame])
        axs[2].set_title(f"Prediction Mask {frame}")
        axs[2].grid(False)

    # Create an animation
    num_slices = data.shape[2]
    anim = animation.FuncAnimation(fig, update, frames=num_slices, interval=100, blit=False)
    if name:
        anim.save(name, fps=10)
    plt.close()

def make_prediction_dataset(dataIrm: DataLoader, roi, device):
    """
    Make prediction on a single CT scan.
    """

    post_pred = Compose([AsDiscrete(argmax=True), KeepLargestConnectedComponent()])

    predicted_masks = []

    for batch in dataIrm:
        inputs, labels = (
            batch["image"].to(device),
            batch["label"].to(device),
        )
        sw_batch_size = 4
        val_outputs = sliding_window_inference(
            inputs, roi, sw_batch_size, model, overlap=0.8, mode="gaussian"
        )

        val_outputs_post = [post_pred(val_output) for val_output in decollate_batch(val_outputs)]
        predicted_masks.append(val_outputs_post)


    return predicted_masks

def make_prediction_get_video(dataIrm, roi, device, name=None):
    """
    Make prediction on a single CT scan and save the video.

    Note: It is not appropriate for fast inference on a dataset. If you
    use a model on cuda, the bottleneck is the generation of the video.
    """

    post_pred = Compose([AsDiscrete(argmax=True), KeepLargestConnectedComponent()])

    inputs, labels = (
        dataIrm["image"].to(device),
        dataIrm["label"].to(device),
    )
    sw_batch_size = 4
    val_outputs = sliding_window_inference(
        inputs, roi, sw_batch_size, model, overlap=0.8, mode="gaussian"
    )

    val_outputs_post = post_pred(val_outputs.cpu().detach().numpy()[0])

    show_irm(inputs.detach().cpu().numpy()[0, 0], labels.cpu().detach().numpy()[0][0], val_outputs_post[0], name=name)




if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Load from checkpoint (created by DistributedDataParallel wrapper)
    model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
    ).to(device)

    # state = torch.load('../model/cleanSlidingWindowCorrected/checkpoint144epochs.pt', map_location=device)
    # state = torch.load('../model/cleanSlidingWindowCorrected2/checkpoint.pt', map_location=device)
    state = torch.load('../model/cleanSlidingWindowCorrected2/checkpoint.pt')
    normal_state = recursive_removal_module(state)
    # load
    model.load_state_dict(normal_state)

    ###
    loader = get_loader("/tsi/data_education/data_challenge/train")

    # loader = get_loader('../data/examples')

    roi = (192, 192, 80)

    with torch.no_grad():
        args_list = make_prediction_dataset(loader, roi, device)
  
    args_list = [(
        img["image"].squeeze(0).squeeze(0).cpu().detach().numpy(),
        img["label"].squeeze(0).squeeze(0).cpu().detach().numpy(),
        arg[0].cpu().detach().numpy(),
        f"video-{i}") 
        for i, (arg, img) in enumerate(zip(args_list, iter(loader)))]

    print(args_list[0][0].shape, args_list[0][1].shape, args_list[0][2][0].shape)

    pool = Pool(6)

    with torch.no_grad():
        pool.starmap(show_irm, args_list)

    pool.close()
    pool.join()
