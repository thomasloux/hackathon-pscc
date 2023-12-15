#!/bin/bash
#SBATCH --partition=IPPMED-A40 # partition (queue)
#SBATCH --job-name=pscc-training-loux
#SBATCH --nodes=1            # total number of nodes
#SBATCH --ntasks-per-node=2  # number of tasks per node
#SBATCH --gres=gpu:2         # number of GPUs reserved per node
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=log/pscc-training%j.out
#SBATCH --error=log/pscc-training%j.err

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

export LOGLEVEL=INFO
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=4 # Arbitrary
port_number=$((20000 + ${SLURM_JOBID}%20000))
echo ${head_node}.enst.fr:${port_number}

 
source ~/.bashrc
source activate pscc

srun  python tutoriel.py \
--total-epochs 100 \
--batch-size 10 \
--data-dir /tsi/data_education/data_challenge/train \
--folder-save model/cleanSlidingWindowCorrected2 \
--save-every 5 \
# --rdzv-id $RANDOM \
# --rdzv-backend c10d \
# --rdzv-endpoint localhost:${port_number} \


echo "Done"
