import os
import json
import shutil
import tempfile
import time
from typing import Tuple, List, Dict, Any, Optional
import glob
from tqdm import tqdm
import pandas as pd

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
    KeepLargestConnectedComponentd,
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
from skimage import measure

from toSubmissionFormat.submission_gen import submission_gen

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

# def find_largest_component(image_data):
#     if image_data.sum() == 0:
#         return np.zeros(image_data.shape)
#     # Find connected components
#     labels = measure.label(image_data, background=0)
#     # Get the largest connected component
#     largest_component = np.zeros(image_data.shape)
#     largest_component[labels == np.argmax(np.bincount(labels.flat)[1:])+1] = 1
#     return largest_component

# def dist(c1, c2):
#     # compute center of mass of each component
#     x1, y1, z1 = np.mean(np.where(c1==1), axis=1)
#     centroid1 = np.array([x1, y1, z1])

#     x2, y2, z2 = np.mean(np.where(c2==1), axis=1)
#     centroid2 = np.array([x2, y2, z2])
#     # weighted distance between centroids because z dimension is shrinked. 
#     pix_dim = np.array([0.9765625, 0.9765625, 3.])
#     return np.linalg.norm((centroid1-centroid2)*pix_dim)

# def merge_connected_components(image_data: np.array, distance: int = 100):

#     # find the 3 largest components
#     component_1= find_largest_component(image_data)
#     component_2= find_largest_component(image_data-component_1)
#     component_3 =find_largest_component(image_data-component_1-component_2)
#     d13=dist(component_1, component_3)
#     d23=dist(component_2, component_3)
#     d12=dist(component_1, component_2)

#     # if the distance between the 3 largest components is less than 100mm, merge them
#     r=distance
#     idx=[1]
#     if d12<r:
#         idx.append(2)
#     if d13<r:
#         idx.append(3)
#     # create a new image with only the components in idx
#     image_data_new=component_1.copy()
#     if 2 in idx:
#         image_data_new+=component_2
#     if 3 in idx:
#         image_data_new+=component_3
#     return image_data_new

# class MergeComponentsd(transforms.MapTransform):
#     def __init__(self, keys: List[str], distance: int = 100):
#         super().__init__(keys)
#         self.distance = distance

#     def __call__(self, data):
#         d = dict(data)
#         for key in self.key_iterator(d):
#             img = d[key].cpu().numpy().squeeze()
#             new_img = merge_connected_components(img, self.distance)
#             d[key] = torch.from_numpy(new_img).unsqueeze(0)
#         return d


def main(
    rank: int,
    world_size: int,
    data_dir: str,
    output_dir: str,
    model_path: str):

    ddp_setup(rank, world_size)
    test_img = sorted(glob.glob(os.path.join(data_dir, "*")))   # Capture .nii or .nii.gz files
    test_data = [{"image": image} for image in test_img]

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            SpatialCropd(keys=["image"], roi_start=(30, 30, 20), roi_end=(512-30, 512-100, 130)),
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
        sampler=DistributedSampler(test_ds, shuffle=False),
    )

    post_transform = Compose(
        [
            Invertd(
                keys="pred",
                transform=test_transform,
                orig_keys="image",
                nearest_interp=True,
                to_tensor=True,
            ),
            AsDiscreted(
                keys="pred",
                argmax=True
                # softmax=True,
            ),
            KeepLargestConnectedComponentd(keys="pred", connectivity=1, num_components=1),
            # FillHolesd(keys="pred", radius=2),
            SaveImaged(keys="pred", output_dir=output_dir, resample=False, output_postfix="", separate_folder=False),
        ]
    )

    roi = (160, 160, 64)
    model = SwinUNETR(
        img_size=roi,
        in_channels=1,
        out_channels=2,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        drop_rate=0.2,
        use_v2=True,
    ).to(rank)
    # model = SwinUNETR(
    #     img_size=roi,
    #     in_channels=1,
    #     out_channels=2,
    #     feature_size=96,
    #     depths=(2, 2, 2, 2, 2),
    #     num_heads=(6, 12, 24, 48, 96),
    #     drop_rate=0.0,
    #     use_v2=True,
    # ).to(rank)

####### Load the model in normal state if it was saved in DistributedDataParallel wrapper
    def recursive_removal_module(input_dict):
        newdict = {}
        for key in input_dict.keys():
            newKey = key[7:]
            if isinstance(input_dict[key], dict):
                newdict[newKey] = recursive_removal_module(input_dict[key])
            else:
                newdict[newKey] = input_dict[key]
        return newdict

    savefile = torch.load(model_path)
    normal_state = recursive_removal_module(savefile)

    model.load_state_dict(normal_state)
#######

    with torch.no_grad():
        for test_data_batch in test_loader:
            test_inputs = test_data_batch["image"].to(rank)
            roi_size = (160, 160, 64)
            sw_batch_size = 12
            test_data_batch["pred"] = sliding_window_inference(
                test_inputs, roi_size, sw_batch_size, model, overlap=0.8, mode="gaussian")

            test_data_post = [post_transform(i) for i in decollate_batch(test_data_batch)]

    # Using the code from Hackathon Organizer to generate the submission file
    if rank == 0:
        submission_gen(output_dir, os.path.join(output_dir, "submission.csv"))

    destroy_process_group()

    # Rewrite the name properly (ex: LUNG1-001)
    # submission = pd.read_csv(os.path.join(output_dir, "submission.csv"))
    # submission["id"] = submission["id"].apply(lambda x: f"LUNG1-{x:03d}")
    # submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/tsi/data_education/data_challenge/test/volume/")
    parser.add_argument("--output-dir", type=str, default="../out/default")
    parser.add_argument("--model-path", type=str)

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    arguments = (world_size, args.data_dir, args.output_dir, args.model_path)
    mp.spawn(main, args=arguments, nprocs=world_size)