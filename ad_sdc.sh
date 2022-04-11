#!/bin/bash

set -e

INJECTIONS=1
DIRECTORY=/tmp/experiment
SHARD=tinyad

mkdir -p $DIRECTORY/home
mkdir -p $DIRECTORY/results
mkdir -p $DIRECTORY/data

docker build -t dsn2022paper165/sc .
#JOBS=`docker run --env SHARD=$SHARD --env INJECTIONS_RANGE="0-$INJECTIONS-1" dsn2022paper165/sc python ./array_count.py`

#docker run -v $DIRECTORY/home/:/root/ -v $DIRECTORY/results/:/code/linearcode/results/ --env SHARD=$SHARD --env INJECTIONS_RANGE="0-$INJECTIONS-1" dsn2022paper165/sc ./pseudo_slurm_map.sh $JOBS
docker run -v $DIRECTORY/home/:/root/ \
           -v $DIRECTORY/results/:/code/linearcode/results/ \
           -v $DIRECTORY/data/:/code/thesis/data/ \
       --env SHARD=$SHARD --env INJECTIONS_RANGE="0-$INJECTIONS-1" dsn2022paper165/sc ./pseudo_slurm_reduce.sh 1
