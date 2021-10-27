#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=12G
#SBATCH --array 0-479

export IMAGENET_ROOT=/home/aasgarik/scratch/data/imagenet
export PYTHONPATH=$PYTHONPATH:/home/aasgarik/projects/def-karthikp/aasgarik/reasilience
source /home/aasgarik/projects/def-karthikp/aasgarik/dnnfault/venv/bin/activate
INTERNAL_SIZE=40
for i in $( eval echo {1..$INTERNAL_SIZE} ); do
  export INTERNAL_SLURM_ARRAY_TASK_ID=$(( SLURM_ARRAY_TASK_ID * INTERNAL_SIZE + i - 1 ))
  echo $INTERNAL_SLURM_ARRAY_TASK_ID
  python map.py
done
