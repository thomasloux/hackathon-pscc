import os
import numpy as np
import torch

from monai.networks.layers import Norm
from monai.networks.nets import UNet

from monai.transforms import Resize

import torch

class SegmentationModel(torch.nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.model = model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
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
