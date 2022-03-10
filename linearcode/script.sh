#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --mem=16G
#SBATCH --array 0-0

export IMAGENET_ROOT=/home/aasgarik/scratch/data/imagenet
export SHARD=quantized

INTERNAL_SIZE=40
for i in $( eval echo {1..$INTERNAL_SIZE} ); do
  export INTERNAL_SLURM_ARRAY_TASK_ID=$(( SLURM_ARRAY_TASK_ID * INTERNAL_SIZE + i - 1 ))
  echo $INTERNAL_SLURM_ARRAY_TASK_ID
  timeout 5m singularity exec --no-home --pwd /code -B ..:/code -B /scratch -B /localscratch ../sc.sif python map.py
done
