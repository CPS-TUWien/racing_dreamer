import argparse
import os
from shutil import copyfile
from time import time

import optuna
from optuna.integration import TensorBoardCallback
from racing.tuning import choose_objective


def main(args):
    timestamp = time()
    logdir = f'logs/tuning/{args.study_name}'

    if args.default_params is not None:
        if not os.path.exists(logdir):
            os.makedirs(logdir, exist_ok=True)
        filename = os.path.basename(args.params).split('.')[0]
        copyfile(src=args.params, dst=f'{logdir}/{filename}_{timestamp}.yml')

    objective_fn = choose_objective(args)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        sampler=optuna.samplers.TPESampler(n_startup_trials=4),
        load_if_exists=True,
        direction='maximize'
    )

    tensorboard_callback = TensorBoardCallback(f'{logdir}/trials', metric_name=args.objective)

    study.optimize(
        func=objective_fn,
        n_trials=args.trials,
        callbacks=[tensorboard_callback],
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning for a specified algorithm.')
    parser.add_argument('--study_name', type=str, required=True)
    parser.add_argument('--agent', type=str, choices=['d4pg', 'mpo', 'sac', 'ppo'], required=True)
    parser.add_argument('--track', type=str, choices=['austria', 'columbia', 'treitlstrasse'], required=True)
    parser.add_argument('--objective', type=str, choices=['max_progress', 'max_speed'], required=True)
    parser.add_argument('--tunable_params', type=str, required=True, help='Path to file containing parameters to tune.')
    parser.add_argument('--steps', type=int, required=True, help='Max. number of steps per trial.')
    parser.add_argument('--trials', type=int, required=False, default=100)
    parser.add_argument('--storage', type=str, required=False)

    parser.add_argument('--action_repeat', type=int, required=False, default=4)
    parser.add_argument('--default_params', type=str, required=False, help='Path to file containing default parameters.')
    args = parser.parse_args()

    if not args.storage:
        args.storage = f'sqlite:///{args.study_name}.db'

    main(args)
