#!/bin/bash
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
export STUDY=$1
export AGENT=$algorithm
export TRACK=austria
export STEPS=100000
export EPOCHS=10
export TRIALS=50
export LOGDIR=$(pwd)/logs/tuning/
docker-compose -f docker/study-compose.yml up  -d --scale tuning=$3 database tuning