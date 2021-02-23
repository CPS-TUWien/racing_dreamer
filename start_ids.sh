#!/bin/bash

args=(
  "austria mpo max_progress 8000000"
  "austria d4pg max_progress 8000000"
  "austria sac max_progress 8000000"
  "austria ppo max_progress 8000000"
  "columbia mpo max_progress 8000000"
  "columbia d4pg max_progress 8000000"
  "columbia sac max_progress 8000000"
  "columbia ppo max_progress 8000000"
  "treitlstrasse_v2 mpo max_progress 8000000"
  "treitlstrasse_v2 d4pg max_progress 8000000"
  "treitlstrasse_v2 sac max_progress 8000000"
  "treitlstrasse_v2 ppo max_progress 8000000"
  "austria lstm-ppo max_progress 8000000"
  "columbia lstm-ppo max_progress 8000000"
  "treitlstrasse_v2 lstm-ppo max_progress 8000000"
)

for exp in "$@"
do
  echo "Running exp_$exp"
  docker container rm "exp_$exp"
  index=$(($exp-1))
  ./start_experiment.sh exp_$exp ${args[$index]}
done
