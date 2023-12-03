import os
import numpy as np
import torch
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import Resize
import nibabel as nib
import torch
import argparse
from model import get_model
from typing import List, Tuple

def arg_parser() -> Tuple[List[str], str, str]:
    ### Argparse ###

    parser = argparse.ArgumentParser(description='Inference script for the PSCC Hackathon')
    parser.add_argument('--input', type=str, help='Input file path or directory to .nii files')
    parser.add_argument('--output', type=str, help='Output path (must be a folder)')
    parser.add_argument('--model', type=str, help='Model weigths file path')
    args = parser.parse_args()

    input = args.input
    output = args.output
    model_weigth_path = args.model
    liste_of_files = []

    ### Check arguments ###
    # Check input
    if os.path.isdir(input):
        input_files = os.listdir(input)
        for file in input_files:
            if not file.endswith(".nii"):
                raise ValueError("Input directory must contain only .nii files")
            liste_of_files.append(os.path.join(input, file))

    if not os.path.isdir(input) and not input.endswith(".nii"):
        raise ValueError("Input must be a directory or a .nii file")
    else:
        liste_of_files.append(input)

    # Check output
    if not os.path.isdir(output):
        os.mkdir(output)

    # Check model
    if not model_weigth_path.endswith(".pth"):
        raise ValueError("Model must be a .pth file")

    return liste_of_files, output, model_weigth_path

def inferrence(input, output, model):
    """
    Create a prediction segmentation mask from a given input image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model().to(device)
    model.load_state_dict(torch.load(model_weigth_path))


    transform_eval = Compose()

    # Load images
    ImageDataset = monai.data.Dataset(data=liste_of_files, transform=transform_eval)
    ImageLoader = DataLoader(ImageDataset, batch_size=1, num_workers=4)

    # Predict
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(ImageLoader):
            inputs = batch[0].to(device)

            # Prediction
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)

            # Thresholding
            outputs = outputs > 0.5
            outputs = outputs.float()

            # Save prediction
            outputs = outputs.detach().cpu().numpy()
            outputs = np.squeeze(outputs)
            outputs = outputs.astype(np.float32)
            outputs = nib.Nifti1Image(outputs, np.eye(4))
            
            @TODO add file name
            nib.save(outputs, os.path.join(output, f"prediction_{i}.nii"))


def main():
    liste_of_files, output, model_weigth_path = arg_parser()
    inferrence(liste_of_files, output, model_weigth_path)

if __name__ == "__main__":
    main()