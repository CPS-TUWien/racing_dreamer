#!/bin/bash
docker container rm exp_1 exp_2 exp_3 exp_4 exp_5 exp_6 exp_7 exp_8 exp_9 exp_10 exp_11 exp_12
./start_experiment.sh exp_1 austria mpo max_progress 8000000
./start_experiment.sh exp_2 austria d4pg max_progress 80000000
./start_experiment.sh exp_3 austria sac max_progress 2000000
./start_experiment.sh exp_4 austria ppo max_progress 2000000
./start_experiment.sh exp_5 columbia mpo max_progress 8000000
./start_experiment.sh exp_6 columbia d4pg max_progress 80000000
./start_experiment.sh exp_7 columbia sac max_progress 2000000
./start_experiment.sh exp_8 columbia ppo max_progress 2000000
./start_experiment.sh exp_9 treitlstrasse_v2 mpo max_progress 8000000
./start_experiment.sh exp_10 treitlstrasse_v2 d4pg max_progress 80000000
./start_experiment.sh exp_11 treitlstrasse_v2 sac max_progress 2000000
./start_experiment.sh exp_12 treitlstrasse_v2 ppo max_progress 2000000
