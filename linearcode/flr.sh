#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=12G
#SBATCH --array 0-100

export IMAGENET_ROOT=/home/aasgarik/scratch/data/imagenet
export PYTHONPATH=$PYTHONPATH:/home/aasgarik/projects/def-karthikp/aasgarik/reasilience
source /home/aasgarik/projects/def-karthikp/aasgarik/dnnfault/venv/bin/activate
INTERNAL_SIZE=60000
INJECTION_SIZE=2
export BATCH_SIZE=1
for i in $( eval echo {1..$INTERNAL_SIZE} ); do
  export _INTERNAL_SLURM_ARRAY_TASK_ID=$(( SLURM_ARRAY_TASK_ID * INTERNAL_SIZE + i - 1 ))
  export INTERNAL_SLURM_ARRAY_TASK_ID=$((_INTERNAL_SLURM_ARRAY_TASK_ID % INJECTION_SIZE))
  export INJECTIONS_RANGE=$((_INTERNAL_SLURM_ARRAY_TASK_ID / INJECTION_SIZE))
  export INJECTIONS_RANGE=$INJECTIONS_RANGE-$((INJECTIONS_RANGE + 1))-'1'
  echo $INJECTIONS_RANGE
  echo $INTERNAL_SLURM_ARRAY_TASK_ID
  python map.py
done
