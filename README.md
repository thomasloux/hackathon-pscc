# Hackathon PSCC
Team : Thomas Loux and Théodore Fougereux

## Download the dataste 

Must be part of the kaggle competition to download the dataset.

```bash
    kaggle competitions download -c copy-of-pscc-data-challenge
```

Then you can configure the path to the dataset for the code to work.
If you are on the IPPMED cluster, the data is available at /tsi/data_education/data_challenge/ 

## Install the requirements

```bash
    pip install -r requirements.txt
```

## Run the code 
The code uses torch multiprocessing to speed up the training with Distributed Data Parallel. It is completely possible to use a single GPU without change. Still you may need to change the batch size to fit your GPU memory.

There are two approaches :
- Whole image processing (monai-whole.py)
- Patch based processing (monai-sliding.py)
You can directly run the code using python or use the bash script (run.sh) to run the code on SLURM.

You can use monai-sliding.py with the weigths from ./cleanSlidingWindowCorrected/checkpoint144epochs.pt

## Inference

You can use video/make_visualization.py to make a video of the 3D scans, real mask and predicted mask. It uses the model to predict the mask, demonstrating the inference pipeline.


