#!/bin/bash
#SBATCH --partition=IPPMED-A40 # partition (queue)
#SBATCH --job-name=pscc-training-loux
#SBATCH --nodes=1            # total number of nodes
#SBATCH --ntasks-per-node=1  # number of tasks per node
#SBATCH --gres=gpu:1         # number of GPUs reserved per node
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=log/pscc-training%j.out
#SBATCH --error=log/pscc-training%j.err

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

export LOGLEVEL=INFO
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=4 # Arbitrary

# If used with conda 
# source ~/.bashrc
# source activate pscc

srun  python main-test.py \
--output-dir out/swinUnetr \
--model-path model/swinUnetr/checkpoint.pt \
--data-dir-test /tsi/data_education/data_challenge/test/volume/ \
--data-dir-train /tsi/data_education/data_challenge/train/volume/ \

echo "Done"
