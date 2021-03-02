import argparse
import os
from functools import partial
import random
from shutil import copyfile
from time import time

import optuna

from racing.experiments.util import read_hyperparams
from racing.tuning.objective import objective



def main(args):
    # load hyperparameters from files
    args.default_params = read_hyperparams(file=args.default_params)
    args.tunable_params = read_hyperparams(file=args.tunable_params)

    # create a parametrized objective function
    objective_fn = partial(objective, args=args)

    # create or load the study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        pruner=optuna.pruners.HyperbandPruner(),
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=True,
        direction='maximize'
    )

    # if the study was just created, do a run with the default parameters
    if len(study.trials) == 0:
        study.enqueue_trial(args.default_params)

    # start the optimization process
    study.optimize(
        func=objective_fn,
        n_trials=args.trials,
        gc_after_trial=True
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning for a specified algorithm.')
    parser.add_argument('--study_name', type=str, required=True)
    parser.add_argument('--track', type=str, choices=['austria', 'columbia', 'treitlstrasse_v2'], required=True)
    parser.add_argument('--task', type=str, choices=['max_progress', 'max_speed'], required=True)
    parser.add_argument('--agent', type=str, choices=['d4pg', 'mpo', 'sac', 'ppo', 'lstm-ppo', 'lstm-mpo'], required=True)
    parser.add_argument('--tunable_params', type=str, required=True, help='Path to file containing parameters to tune.')
    parser.add_argument('--default_params', type=str, required=True, help='Path to file containing parameters to tune.')
    parser.add_argument('--steps', type=int, required=True, help='Max. number of steps per trial.')
    parser.add_argument('--epochs', type=int, required=True, help='Max. number of steps per trial.')
    parser.add_argument('--trials', type=int, required=False, default=100)
    parser.add_argument('--storage', type=str, required=False)
    parser.add_argument('--training_time_limit', type=int, required=False, default=2000)
    parser.add_argument('--eval_time_limit', type=int, required=False, default=4000)
    parser.add_argument('--eval_interval', type=int, required=False, default=10_000)
    parser.add_argument('--action_repeat', type=int, required=False, default=4)
    parser.add_argument('--logdir', type=str, default='logs/tuning')
    parser.add_argument('--seed', type=int, required=False, default=random.randint(0, 100_000_000))

    args = parser.parse_args()
    main(args)
