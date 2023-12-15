import os
import glob

import nibabel as nib
import pandas as pd

import argparse

from hackathon.post_proc import mask2rle, find_largest_containing_circle


def submission_gen(predpath: list, outputpath: str):
    """Create a submission csv from a path of segmentation prediction.
    predpath: Path of your fodler containing the predictions
    /!\ : The path should directly contain the .nii.gz files
    outputpath: Path of where the csv will be saved out"""
    pred_files = glob.glob(f"{predpath}/*/*")
    rle_list = []
    recist_list = []
    volume_list = []
    patient_id_list = []
    shape_list = []
    for file in sorted(pred_files):
        print(f"Encoding {file}...")
        img = nib.load(file)
        data = img.get_fdata()
        shape_list.append(data.shape)
        rle_list.append(mask2rle(data))
        (
            recist,
            predicted_volume,
            largest_circle,
            largest_slice,
        ) = find_largest_containing_circle(data, img.header["pixdim"][1:4])
        if recist < 0:
            recist = 0
        recist_list.append(recist)
        volume_list.append(predicted_volume)
        filename = file.split("/")[-1]
        patient_id_list.append(filename.split(".")[0])
    df = pd.DataFrame(
        {
            "id": patient_id_list,
            "rle": rle_list,
            "recist": recist_list,
            "volume": volume_list,
            "data_shape": shape_list,
        }
    )
    df.to_csv(outputpath, index=False)
    return f"submission file saved at {outputpath}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Submission_gen",
        description="Convert a list of nifti Nifti files into the appropriate submission file for the IPPMed 2024 data challenge",
    )
    parser.add_argument(
        "pred_path",
        help="path of the nifti files",
    )
    parser.add_argument(
        "submission_filename",
        type=str,
        help="path of the submission file",
    )
    args = parser.parse_args()
    submission_gen(args.pred_path, args.submission_filename)
    print(f"Submission file saved at {args.submission_filename}")
