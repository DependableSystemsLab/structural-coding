#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32G
#SBATCH --array 1-1151

export PYTHONPATH=/home/aasgarik/projects/def-karthikp/aasgarik/reasilience
source /home/aasgarik/projects/def-karthikp/aasgarik/dnnfault/venv/bin/activate
python sga.py
