from typing import Dict

from optuna import Trial


def get_params(trial: Trial, tunable_params: Dict, default_params: Dict):
    defaults = default_params.copy()
    for key in tunable_params:
        args = tunable_params[key]
        defaults[key] = trial.suggest_float(name=key, **args)
    return defaults



