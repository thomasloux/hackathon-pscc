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
    ToNumpyd,
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

import nibabel as nib
import numpy as np
import os
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation

def get_data_seg(id):
    # get rle from ith row of df
    rle = df.iloc[id-1,1]
    # get shape from ith row of df
    shape = df.iloc[id-1,4]
    # transform shape from str (x,y,z) to tuple (x,y,z)
    shape = tuple(map(int, shape[1:-1].split(',')))
    print(shape)
    data= rle2mask(rle, shape)
    return data

def get_data_vol(id):
    id = str(id).zfill(3)
    path = "C:/Users/todof/OneDrive/Documents/Polytechnique/Hackaton telecom/data/data/test/volume/LUNG1-{}_vol.nii.gz".format(id)
    # load the image from path
    img = nib.load(path)
    # get the image data
    data = img.get_fdata()
    affine= img.affine
    # get pixdim
    pixdim = img.header['pixdim']
    return data, pixdim, affine


class dilationErosionPostProcessd(transforms.MapTransform):
    """
    Try to add or remove a layer of pixels from the mask that followed closely the intensity of the image in the same region
    """
    def __init__(self, keys: List[str]):
        super().__init__(keys)
        self.keys = keys
        # Assume keys = ["pred", "image"] or equivalent keys

    def __call__(self, data):
        d = dict(data)
        return self.postProcess(d)

    def postProcess(self, data_dictionary):
        data = data_dictionary[self.keys[0]].squeeze(0)
        dilated_data = binary_dilation(data).astype(data.dtype)
        eroded_data= binary_erosion(data).astype(data.dtype)

        data_vol = data_dictionary[self.keys[1]].squeeze(0)
        data_vol = (data_vol - np.min(data_vol))/(np.max(data_vol)-np.min(data_vol))

        i_mean= np.mean(data_vol[data==1].flatten())
        i_std= np.std(data_vol[data==1].flatten())
    
        new_data= data.copy()
        new_data[(eroded_data==0) &  (data==1) & (abs(data_vol-i_mean)>4*i_std)]=0
        new_data[(dilated_data==1) &  (data==0) & (abs(data_vol-i_mean)<0.1*i_std)]=1

        # if new data is empty, make the ball with radius 10 and center 50, 50, 50  as 1
        if np.sum(new_data)==0:
            print(new_data.shape)
            new_data[250,250,50]=1
            new_data= binary_dilation(new_data, iterations=10).astype(new_data.dtype)
        return {self.keys[0]: np.expand_dims(new_data, axis=0), self.keys[1]: data_dictionary[self.keys[1]]}

class SpatialCropCustomd(transforms.MapTransform):
    """
    Crop differently if the image is long or short (in the z axis)
    """
    def __init__(
        self,
        roi_start_small: Tuple[int, int, int],
        roi_end_small: Tuple[int, int, int],
        roi_start_large: Tuple[int, int, int],
        roi_end_large: Tuple[int, int, int],
        size_threshold: int,
        keys: List[str]):
        super().__init__(keys)
        self.keys = keys
        self.size_threshold = size_threshold
        self.crop_small = SpatialCropd(keys=keys, roi_start=roi_start_small, roi_end=roi_end_small)
        self.crop_large = SpatialCropd(keys=keys, roi_start=roi_start_large, roi_end=roi_end_large)

    def __call__(self, data):
        shape = data[self.keys[0]].shape
        if shape[-1] > self.size_threshold:
            for key in self.key_iterator(data):
                data = self.crop_large(data)
        else:
            for key in self.key_iterator(data):
                data = self.crop_small(data)
        return data


   

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

def main(
    rank: int,
    world_size: int,
    data_dir_test: str,
    data_dir_train: str,
    output_dir: str,
    model_path: str):

    ddp_setup(rank, world_size)
    test_img = sorted(glob.glob(os.path.join(data_dir_test, "*")))   # Capture .nii or .nii.gz files
    test_data = [{"image": image} for image in test_img][:2]
    train_img = sorted(glob.glob(os.path.join(data_dir_train, "*")))   # Capture .nii or .nii.gz files
    train_data = [{"image": image} for image in train_img][:2]

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            SpatialCropCustomd(
                keys=["image"],
                roi_start_small=(70, 70, 0),
                roi_end_small=(512-60, 512-110, 130),
                roi_start_large=(70, 70, 70),
                roi_end_large=(512-60, 512-110, 200),
                size_threshold=250
            ),
            # transforms.CropForegroundd(
            #     keys=["image"],
            #     source_key="image"
            # ),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0.0, b_max=1.0, clip=True),
        ]
    )

    test_ds = Dataset(data=test_data, transform=test_transform)
    train_ds = Dataset(data=train_data, transform=test_transform)

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=4,
        sampler=DistributedSampler(test_ds, shuffle=False),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        num_workers=4,
        sampler=DistributedSampler(train_ds, shuffle=False),
    )

    post_transform_test = Compose(
        [
            Invertd(
                keys=["pred", "image"],
                transform=test_transform,
                orig_keys="image",
                nearest_interp=True,
                to_tensor=True,
            ),
            AsDiscreted(
                keys="pred",
                argmax=True
            ),
            KeepLargestConnectedComponentd(keys="pred", connectivity=1, num_components=1),
            ToNumpyd(keys=["pred", "image"]),
            dilationErosionPostProcessd(keys=["pred", "image"]),
            # ifEmptyPostProcessd(keys=["pred"]),
            SaveImaged(keys="pred", output_dir=os.path.join(output_dir, 'test'), resample=False, output_postfix="", separate_folder=False),
        ]
    )

    post_transform_train = Compose(
        [
            Invertd(
                keys=["pred", "image"],
                transform=test_transform,
                orig_keys="image",
                nearest_interp=True,
                to_tensor=True,
            ),
            AsDiscreted(
                keys="pred",
                argmax=True
            ),
            KeepLargestConnectedComponentd(keys="pred", connectivity=1, num_components=1),
            ToNumpyd(keys=["pred", "image"]),
            dilationErosionPostProcessd(keys=["pred", "image"]),
            # ifEmptyPostProcessd(keys=["pred"]),
            SaveImaged(keys="pred", output_dir=os.path.join(output_dir, 'train'), resample=False, output_postfix="", separate_folder=False),
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

            test_data_post = [post_transform_test(i) for i in decollate_batch(test_data_batch)]

    with torch.no_grad():
        for train_data_batch in train_loader:
            train_inputs = train_data_batch["image"].to(rank)
            roi_size = (160, 160, 64)
            sw_batch_size = 12
            train_data_batch["pred"] = sliding_window_inference(
                train_inputs, roi_size, sw_batch_size, model, overlap=0.8, mode="gaussian")

            train_data_post = [post_transform_train(i) for i in decollate_batch(train_data_batch)]

    # Using the code from Hackathon Organizer to generate the submission file
    if rank == 0:
        # Renmame output files
        files = os.listdir(os.path.join(output_dir, "test"))
        for file in files:
            index = int(file.split(".")[0])
            os.rename(os.path.join(output_dir, "test", file), os.path.join(output_dir, "test", f"LUNG1-{index:03d}.nii.gz"))

        files = os.listdir(os.path.join(output_dir, "train"))
        for file in files:
            index = int(file.split(".")[0])
            os.rename(os.path.join(output_dir, "train", file), os.path.join(output_dir, "train", f"LUNG1-{index:03d}.nii.gz"))

        submission_gen(os.path.join(output_dir, "test"), os.path.join(output_dir, "test", "submission.csv"))
        submission_gen(os.path.join(output_dir, "train"), os.path.join(output_dir, "train", "submission.csv"))


    destroy_process_group()

    # Rewrite the name properly (ex: LUNG1-001)
    # submission = pd.read_csv(os.path.join(output_dir, "submission.csv"))
    # submission["id"] = submission["id"].apply(lambda x: f"LUNG1-{x:03d}")
    # submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir-test", type=str, default="/tsi/data_education/data_challenge/test/volume/")
    parser.add_argument("--data-dir-train", type=str, default="/tsi/data_education/data_challenge/train/volume/")
    parser.add_argument("--output-dir", type=str, default="../out/default")
    parser.add_argument("--model-path", type=str)

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    arguments = (world_size, args.data_dir_test, args.data_dir_train, args.output_dir, args.model_path)
    mp.spawn(main, args=arguments, nprocs=world_size)