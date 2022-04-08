#!/bin/bash

COUNT=$1

for i in $( eval echo {0..$COUNT} ); do
  export INTERNAL_SLURM_ARRAY_TASK_ID=$i
  timeout 5m python map.py
done