#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32G
#SBATCH --array 0-3600

export PYTHONPATH=/home/aasgarik/projects/def-karthikp/aasgarik/reasilience
source /home/aasgarik/projects/def-karthikp/aasgarik/dnnfault/venv/bin/activate
INTERNAL_SIZE=20
for i in $( eval echo {1..$INTERNAL_SIZE} ); do
  export INTERNAL_SLURM_ARRAY_TASK_ID=$(( SLURM_ARRAY_TASK_ID * INTERNAL_SIZE + i - 1 ))
  echo $INTERNAL_SLURM_ARRAY_TASK_ID
  python sga.py
done
