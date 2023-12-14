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
    AsDiscreted,
    Activations,
    ScaleIntensityRanged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRange,
    RandCropByPosNegLabeld,
    Compose,
    SpatialCropd,
    Invertd,
    SaveImaged,
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



test_img = sorted(glob.glob("/tsi/data_education/data_challenge/test/volume/*.nii.gz"))
test_data = [{"image": image} for image in test_img]

test_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        SpatialCropd(keys=["image"], roi_start=(30, 30, 0), roi_end=(512-30, 512-100, 130)),
        transforms.CropForegroundd(
            keys=["image"],
            source_key="image"
        ),
        Orientationd(keys=["image"], axcodes="RAS"),
        # Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        # Probably not needed 
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0.0, b_max=1.0, clip=True),
    ]
)

test_ds = Dataset(data=test_data, transform=test_transform)

test_loader = DataLoader(
    test_ds,
    batch_size=1,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)

post_transform = Compose(
    [
        Invertd(
            keys="pred",
            transform=test_transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            nearest_interp=True,
            to_tensor=True,
        ),
        AsDiscreted(
            keys="pred",
            argmax=True,
            to_onehot=2
        ),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out", output_postfix="PredSeg", resample=False),
    ]
)

device = torch.device("cuda:0")

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
    dropout=0.2,
).to(device)

def recursive_removal_module(input_dict):
    newdict = {}
    for key in input_dict.keys():
        newKey = key[7:]
        if isinstance(input_dict[key], dict):
            newdict[newKey] = recursive_removal_module(input_dict[key])
        else:
            newdict[newKey] = input_dict[key]
    return newdict

# savefile = torch.load("model/largeRoi/checkpoint.pt")
savefile = torch.load("model/cleanSlidingWindowCorrected/checkpoint.pt")
normal_state = recursive_removal_module(savefile)

model.load_state_dict(normal_state)

with torch.no_grad():
    for test_data_batch in test_loader:
        test_inputs = test_data_batch["image"].to(device)
        # roi_size = (192, 192, 64)
        roi_size = (128, 128, 64)
        sw_batch_size = 4
        test_data_batch["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model, overlap=0.7)

        test_data_post = [post_transform(i) for i in decollate_batch(test_data_batch)]