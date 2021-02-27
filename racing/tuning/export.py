import argparse

import pandas
import optuna

def main(args):
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    df = study.trials_dataframe()
    df.to_csv(path_or_buf=f'{args.path}/{args.study_name}.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning for a specified algorithm.')
    parser.add_argument('--study_name', type=str, required=True)
    parser.add_argument('--storage', type=str, required=False)
    parser.add_argument('--path', type=str, default='.')
    args = parser.parse_args()

    if not args.storage:
        args.storage = f'mysql+pymysql://user:password@localhost/{args.study_name}'

    main(args)