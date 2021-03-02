#!/bin/bash

args=(
  "austria mpo max_progress 8000000 hyperparams/default/mpo.yml"
  "austria d4pg max_progress 8000000 hyperparams/default/d4pg.yml"
  "austria sac max_progress 8000000 hyperparams/default/sac.yml"
  "austria ppo max_progress 8000000 hyperparams/default/ppo.yml"
  "columbia mpo max_progress 8000000 hyperparams/default/mpo.yml"
  "columbia d4pg max_progress 8000000 hyperparams/default/d4pg.yml"
  "columbia sac max_progress 8000000 hyperparams/default/sac.yml"
  "columbia ppo max_progress 8000000 hyperparams/default/ppo.yml"
  "treitlstrasse_v2 mpo max_progress 8000000 hyperparams/default/mpo.yml"
  "treitlstrasse_v2 d4pg max_progress 8000000 hyperparams/default/d4pg.yml"
  "treitlstrasse_v2 sac max_progress 8000000 hyperparams/default/sac.yml"
  "treitlstrasse_v2 ppo max_progress 8000000 hyperparams/default/ppo.yml"
  "austria lstm-ppo max_progress 8000000 hyperparams/default/ppo-lstm.yml"
  "columbia lstm-ppo max_progress 8000000 hyperparams/default/ppo-lstm.yml"
  "treitlstrasse_v2 lstm-ppo max_progress 8000000 hyperparams/default/ppo-lstm.yml"
)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $DIR
for exp in "$@"
do
  echo "Running exp_$exp"
  docker container rm "exp_$exp"
  index=$(($exp-1))
  $DIR/start_experiment.sh exp_$exp $(echo ${args[$index]})
done
