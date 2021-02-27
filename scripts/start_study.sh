#!/bin/bash
export LABEL=sb3
export STUDY=teststudy
export AGENT=ppo
export TRACK=austria
export STEPS=1000
export EPOCHS=2
export TRIALS=2
export LOGDIR=$(pwd)/logs/tuning/
docker-compose -f docker/study-compose.yml up  -d --scale tuning=$1 database tuning