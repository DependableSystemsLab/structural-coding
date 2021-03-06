#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --mem=16G
#SBATCH --array 0-40

module load singularity
mkdir -p ../shome
mkdir -p results

export SHARD=resnet50coverage

INTERNAL_SIZE=40
for i in $( eval echo {1..$INTERNAL_SIZE} ); do
  export INTERNAL_SLURM_ARRAY_TASK_ID=$(( SLURM_ARRAY_TASK_ID * INTERNAL_SIZE + i - 1 ))
  echo $INTERNAL_SLURM_ARRAY_TASK_ID
  timeout 5m singularity exec --no-home --pwd /code/linearcode -B ../shome:/home/aasgarik -B results/:/code/linearcode/results/ ../sc.sif python map.py
done
