#!/bin/bash
set -e

docker build -t dsn2022paper165/sc .
docker push dsn2022paper165/sc
singularity build sc.sif docker://dsn2022paper165/sc:latest
scp sc.sif aasgarik@cedar.computecanada.ca:/home/aasgarik/projects/def-karthikp/aasgarik/reasilience/sc.sif
