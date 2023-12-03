import os
import numpy as np
import torch

from monai.networks.layers import Norm
from monai.networks.nets import UNet

from monai.transforms import Resize

import torch

def get_model():
    """
    Get model with the best hyperparameters.
    """

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    return model