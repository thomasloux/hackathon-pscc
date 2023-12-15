# Hackathon PSCC
Team : Thomas Loux and Théodore Fougereux

## Download the dataste 

Must be part of the kaggle competition to download the dataset.

```bash
    kaggle competitions download -c copy-of-pscc-data-challenge
```
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



