# Build the Dataset for the project
# Using Monai Dataset class

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    Resize,
    ScaleIntensityRange,
    ToTensor,
)
from monai.data import ImageDataset
import nibabel as nib


class LungCancerDataset(Dataset):
    def __init__(self, root, root_processed, transform_img=None, transform_seg=None, seg_dir="seg", image_dir="volume"):
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.transform_img = transform_img
        self.transform_seg = transform_seg
        self.root = root

        if not os.path.exists(root):
            self._download()
        if not os.path.isdir(root):
            raise ValueError("root must be a directory")

        self.parent_dir_root = root_processed
        os.path.dirname(os.path.dirname(self.root))
        self.path_preprocessed = os.path.join(self.parent_dir_root, "preprocessed")

        if not os.path.exists(os.path.dirname(self.path_preprocessed)) or not os.path.exists(self.path_preprocessed):
            if not os.path.exists(os.path.dirname(self.path_preprocessed)):
                os.mkdir(os.path.dirname(self.path_preprocessed))
            os.mkdir(self.path_preprocessed)
            os.mkdir(os.path.join(self.path_preprocessed, self.image_dir))
            os.mkdir(os.path.join(self.path_preprocessed, self.seg_dir))
            self._preprocess()

        self.image_files = os.listdir(os.path.join(self.path_preprocessed, self.image_dir))
        self.seg_files = os.listdir(os.path.join(self.path_preprocessed, self.seg_dir))

        self.image_files.sort()
        self.seg_files.sort()


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = nib.load(os.path.join(self.path_preprocessed, self.image_dir, self.image_files[idx])).get_fdata()
        mask = nib.load(os.path.join(self.path_preprocessed, self.seg_dir, self.seg_files[idx])).get_fdata()
        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_seg:
            mask = self.transform_seg(mask)
        return image, mask

    def _download(self):
        """
        Download the dataset
        """
        raise NotImplementedError("Need to download it first or check the path")

    def _preprocess(self):
        """
        Preprocess the dataset
        """
        # Transformations
        preprocess_image = Compose(
            [
                lambda x: x[np.newaxis, :, :, :],
                Resize((512, 512, 128)),
                ScaleIntensityRange(a_min=-1024, a_max=3071, b_min=0.0, b_max=1.0, clip=True),
                lambda x: x.detach().cpu().numpy()
            ]
        )
        preprocess_seg = Compose(
            [
                lambda x: x[np.newaxis, :, :, :],
                Resize((512, 512, 128)),
                lambda x: x.detach().cpu().numpy()
            ]
        )
        # Path to raw files
        image_files = os.listdir(os.path.join(self.root, self.image_dir))
        seg_files = os.listdir(os.path.join(self.root, self.seg_dir))

        # # Transform and save
        # for image_file in image_files:
        #     image = nib.load(os.path.join(self.root, self.image_dir, image_file)).get_fdata()
        #     image = preprocess_image(image)
        #     image = nib.Nifti1Image(image, np.eye(4))
        #     nib.save(image, os.path.join(self.path_preprocessed, self.image_dir, image_file))

        # for seg_file in seg_files:
        #     mask = nib.load(os.path.join(self.root, self.seg_dir, seg_file)).get_fdata()
        #     mask = preprocess_seg(mask)
        #     mask = nib.Nifti1Image(mask, np.eye(4))
        #     nib.save(mask, os.path.join(self.path_preprocessed, self.seg_dir, seg_file))
        dataset = ImageDataset(
            image_files=[os.path.join(self.root, self.image_dir, image_file) for image_file in image_files],
            seg_files=[os.path.join(self.root, self.seg_dir, seg_file) for seg_file in seg_files],
            transform=preprocess_image,
            seg_transform=preprocess_seg
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=4
        )
        for i, batch in enumerate(loader):
            image = batch[0]
            mask = batch[1]
            image = image.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
            for j in range(image.shape[0]):
                image_nib = nib.Nifti1Image(image[j], np.eye(4))
                mask_nib = nib.Nifti1Image(mask[j], np.eye(4))
                nib.save(image_nib, os.path.join(self.path_preprocessed, self.image_dir, f"image_{i*image.shape[0]+j}.nii"))
                nib.save(mask_nib, os.path.join(self.path_preprocessed, self.seg_dir, f"mask_{i*image.shape[0]+j}.nii"))


