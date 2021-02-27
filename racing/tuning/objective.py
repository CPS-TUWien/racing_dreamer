import numpy as np
import optuna
from optuna import Trial
from racing.experiments import dispatch_experiment
from racing.tuning.util import get_params


def objective(trial: Trial, args):
    args.params = get_params(trial=trial, tunable_params=args.tunable_params, default_params=args.default_params)
    experiment, agent_ctor = dispatch_experiment(args, f'{args.logdir}/{trial.number}')
    agent = experiment.configure_agent(agent_ctor)
    objective_value = -1e9
    for epoch in range(1, args.epochs+1):
        objective_value = experiment.run_trial(agent=agent, steps=args.steps)
        print(f'Trial-{trial.number}: {objective_value} in epoch {epoch}')
        if not np.isnan(objective_value):
            trial.report(objective_value, step=epoch)

        if trial.should_prune():
            print(f'Trial {trial.number}: pruning.')
            raise optuna.exceptions.TrialPruned()

    if np.isnan(objective_value):
        return -1e9
    else:
        return objective_value

