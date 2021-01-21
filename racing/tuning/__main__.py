import argparse
import os
from typing import Callable

import optuna
from optuna.integration import TensorBoardCallback


def objective(algorithm: str) -> Callable:
    if algorithm == 'mpo':
        pass
    elif algorithm == 'd4pg':
        pass
    else:
        raise NotImplementedError(algorithm)


def main(args):

    if args.resume:
        study = optuna.load_study(
            study_name=args.study_name,
            storage="postgresql://{}:{}@postgres:5432/{}".format(
                os.environ["POSTGRES_USER"],
                os.environ["POSTGRES_PASSWORD"],
                os.environ["POSTGRES_DB"],
            ),
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage="postgresql://{}:{}@postgres:5432/{}".format(
                os.environ["POSTGRES_USER"],
                os.environ["POSTGRES_PASSWORD"],
                os.environ["POSTGRES_DB"],
            ),
        )

    logdir = f'logs/tuning/{args.study_name}/trials'
    tensorboard_callback = TensorBoardCallback(logdir, metric_name='avg_return')

    objective_fn = objective(algorithm=args.agent)
    study.optimize(
        func=objective_fn,
        n_trials=args.n_trials,
        callbacks=[tensorboard_callback],
        catch=(ValueError,),
        gc_after_trial=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a single agent experiment optimizing the progress based reward.')
    parser.add_argument('--study_name', type=str, required=True)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--agent', type=str, choices=['mpo', 'd4pg'], required=True)
    parser.add_argument('--n_trials', type=int, required=True)
    parser.add_argument('--tracks', nargs='+', choices=['austria', 'columbia', 'treitlstrasse'])


    args = parser.parse_args()
    print(args.tracks)
    main(args)