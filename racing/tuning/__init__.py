from typing import Callable

from optuna import Trial

def objective(trial):
    x = trial.suggest_uniform("x", -10., 10.)
    y = trial.suggest_uniform("y", -5., 5.)
    return x ** 2. + y



def choose_objective(args) -> Callable[[Trial], float]:
    if args.agent in ['mpo', 'd4pg']:
        return objective
    elif args.agent in ['sac', 'ppo']:
        pass
    else:
        raise NotImplementedError(args.agent)