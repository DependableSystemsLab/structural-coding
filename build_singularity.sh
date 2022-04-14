#!/bin/bash
set -e

SLURM_WORKING_DIRECTORY=aasgarik@cedar.computecanada.ca:/home/aasgarik/projects/def-karthikp/aasgarik/reasilience/

singularity build sc.sif docker://dsn2022paper165/sc:latest
scp sc.sif $SLURM_WORKING_DIRECTORY/sc.sif
scp linearcode/script.sh $SLURM_WORKING_DIRECTORY/script.sh
