#!/bin/bash
export STUDY=$1
export LOGDIR=$(pwd)/logs/tuning
docker-compose -f docker/study-compose.yml up report