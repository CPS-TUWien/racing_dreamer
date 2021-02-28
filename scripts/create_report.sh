#!/bin/bash
STUDY=$1
LOGDIR=$(pwd)/logs/tuning
docker run --rm --network host -v $LOGDIR/$STUDY:/app/evaluations axel/racing:sb3 python racing/tuning/evaluation.py --study_name ${STUDY}