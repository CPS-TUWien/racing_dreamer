import argparse
import os

import pandas
import yaml
import optuna
from optuna.visualization import plot_intermediate_values, plot_optimization_history, plot_parallel_coordinate, \
    plot_slice, plot_param_importances


def make_plots(logdir, study):
    logdir = f'{logdir}/plots'
    os.makedirs(logdir, exist_ok=True)
    plot_optimization_history(study).write_image(f'{logdir}/history.svg')
    plot_intermediate_values(study).write_image(f'{logdir}/intermediates.svg')
    plot_parallel_coordinate(study).write_image(f'{logdir}/parallel_coordinates.png')
    plot_slice(study).write_image(f'{logdir}/slices.svg')
    plot_param_importances(study).write_image(f'{logdir}/importances.svg')

def write_data(logdir, study):
    study.trials_dataframe().to_csv(f'{logdir}/trials.csv')
    with open(f'{logdir}/params.yml', 'w') as file:
        results = dict(
            best_params=study.best_params,
            objective_value=study.best_value
        )
        yaml.safe_dump(results, file)

def main(args):
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    os.makedirs(args.logdir, exist_ok=True)
    make_plots(logdir=args.logdir, study=study)
    write_data(logdir=args.logdir, study=study)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning for a specified algorithm.')
    parser.add_argument('--study_name', type=str, required=True)
    parser.add_argument('--storage', type=str, required=False)
    parser.add_argument('--logdir', type=str, default='evaluations/')
    args = parser.parse_args()

    if not args.storage:
        args.storage = f'mysql+pymysql://user:password@localhost/{args.study_name}'

    main(args)