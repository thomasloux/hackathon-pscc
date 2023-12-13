#!/bin/bash
#SBATCH --partition=IPPMED-P100 # partition (queue)
#SBATCH --job-name=pscc-training-loux
#SBATCH --nodes=1            # total number of nodes
#SBATCH --ntasks-per-node=3  # number of tasks per node
#SBATCH --gres=gpu:3         # number of GPUs reserved per node
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=log/pscc-training%j.out
#SBATCH --error=log/pscc-training%j.err

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=4 # Arbitrary
 
source ~/.bashrc
source activate pscc

srun torchrun \
--nnodes 1 \
--nproc-per-node 3 \
--rdzv-id $RANDOM \
--rdzv-backend c10d \
--rdzv-endpoint "$head_node_ip:21430" \
monai-ddp-torchrun.py --epochs 5 --folder-save test

echo "Done"
