# Zorg team submission

Install dependencies:
```
pip install -r requirements.txt
```

Note: you may install conda and use conda to install the dependencies.

Run the training script:
```
sbatch run-training.sh
```
You can should uncomment the line 
```
# source ~/.bashrc
# source activate pscc
```
if you are using conda in a bash shell.

The training script will run the training for 400 epochs and save the model weights in the `models` folder.
By default the data folder is the same as for the rest of the hackathon.

Run the evaluation script:
```
sbatch run-inference.sh
```
It will create the binary masks for both training and test data and save them in the `out` folder.

