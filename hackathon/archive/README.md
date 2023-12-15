Files that are not used anymore but are kept for reference.

## Torchrun

I decided to stop using torchrun because it was unstable with ConnectionError with RendezVous framework. I had this problem with multinode training AND also mono mode training. So I decided to only use torch multiprocessing for mono node training.

## Model

I dropped this file because I wanted at the beginning to use binary classification and add the prediction class directly with a function predict. But Monai encourages to use a model with raw output and use a complete postprocessing pipeline (for instance to only keep the largest connected component). So I decided to drop this file and use the model directly in the training script.