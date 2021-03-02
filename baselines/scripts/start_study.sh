#!/bin/bash

study=$1
algorithm=$2
instances=$3

algorithm=$2
if [[ $algorithm == 'mpo' ]] || [[ $algorithm == 'd4pg' ]]
then
    tag=acme
elif [[ $algorithm == 'sac' ]] || [[ $algorithm == 'ppo' ]]
then
    tag=sb3
elif [[ $algorithm == 'lstm-ppo' ]]
then
    tag=sb2
else
  echo "Not a valid algorithm."
  exit 1
fi

export LABEL=$tag
export STUDY=$study
export AGENT=$algorithm
export TRACK=austria
export STEPS=100000
export EPOCHS=10
export TRIALS=50
export LOGDIR=$(pwd)/logs/tuning/$study

echo $tag
echo $study
echo $algorithm
echo $LOGDIR
docker-compose -f docker/study-compose.yml up -d database
sleep 10
docker-compose -f docker/study-compose.yml up -d tuning
sleep 1
docker-compose -f docker/study-compose.yml up -d --scale tuning=$instances database tuning