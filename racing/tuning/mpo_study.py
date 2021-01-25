import functools
import time
from typing import List

import optuna
from acme import make_environment_spec
from optuna import Trial
from optuna.integration import TensorBoardCallback
from racecar_gym import SingleAgentScenario, SingleAgentRaceEnv
import numpy as np
from racing.algorithms import make_mpo_agent
from racing.experiments.acme.logger import TensorBoardLogger, PrefixedTensorBoardLogger
from racing.environment import wrap_env
from racing import experiments

def objective(trial: Trial, env_steps: int, test_interval: int, test_episodes: int, tracks: List[str], action_repeat: int = 1):
    scenarios = [SingleAgentScenario.from_spec(f'scenarios/{track}.yml') for track in tracks]
    train_env = SingleAgentRaceEnv(scenarios[0])
    test_env = SingleAgentRaceEnv([SingleAgentScenario.from_spec(f'scenarios/{track}.yml') for track in tracks][0])
    train_env = wrap_env(env=train_env, wrapper_configs='./configs/environment/single_agent_wrappers.yml')
    test_env = wrap_env(env=test_env, wrapper_configs='./configs/environment/single_agent_test_wrappers.yml')

    tunable_params = dict(
        loss_epsilon=trial.suggest_float(name='loss_epsilon', low=0.001, high=0.1),
        loss_epsilon_mean=trial.suggest_float(name='loss_epsilon_mean', low=0.0001, high=0.01),
        loss_epsilon_stddev=trial.suggest_float(name='loss_epsilon_stddev', low=1e-7, high=1e-5),
        discount=trial.suggest_float(name='discount', low=0.95, high=1.0),
        policy_lr=trial.suggest_float(name='policy_lr', low=1e-5, high=1e-1),
        critic_lr=trial.suggest_float(name='policy_lr', low=1e-5, high=1e-1),
        checkpoint=False
    )

    logger = TensorBoardLogger(logdir=f'hyperparams/{trial.study.study_name}/runs/run-{trial.number}')
    algorithm_logger = PrefixedTensorBoardLogger(base_logger=logger, prefix='algorithm')
    train_logger = PrefixedTensorBoardLogger(base_logger=logger, prefix='training')
    test_logger = PrefixedTensorBoardLogger(base_logger=logger, prefix='test')
    agent = make_mpo_agent(env_spec=make_environment_spec(train_env), logger=algorithm_logger, hyperparams=tunable_params)
    objective_value = -1e9
    step = 0
    try:
        print(f'trial-{trial.number}: start train-test loop')
        while step < env_steps:
            try:
                print(f'trial-{trial.number}: start training')
                train_summary = experiments.train(agent=agent, steps=test_interval, env=train_env, action_repeat=action_repeat, logger=train_logger, current_step=step)
                print(f'trial-{trial.number}: start evaluation')
                test_summary = experiments.test(agent=agent, env=test_env, episodes=test_episodes, action_repeat=action_repeat)
            except KeyError as e:
                print(e.args)
                return -1e9
            step += train_summary['steps']
            print(f'trial-{trial.number}: finished evaluation. logging. step: {step}')

            test_logger.write(test_summary, step=step)

            objective_value = test_summary['avg_episode_return']
            if not np.isnan(objective_value):
                trial.report(objective_value, step=step)
            if trial.should_prune():
                print(f'trial-{trial.number}: pruning.')
                train_env.close()
                test_env.close()
                logger.close()
                raise optuna.exceptions.TrialPruned()
    except ValueError as e:
        print(f'trial-{trial.number}: Value error: {e.args}')
    print(f'trial-{trial.number}: finished optimization.')
    train_env.close()
    test_env.close()
    logger.close()
    if np.isnan(objective_value):
        return -1e9
    return objective_value


def mpo_study(track: str):
    name = f'mpo_hp_study_{time.time()}'
    storage_name ='mysql+pymysql://user:password@localhost:3306/optuna'
    tensorboard_callback = TensorBoardCallback(f"hyperparams/{name}/trials", metric_name="avg_return")

    study = optuna.create_study(
        study_name=name,
       # storage=storage_name,
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        sampler=optuna.samplers.TPESampler(n_startup_trials=4),
        direction='maximize'
    )
    objective_fn = functools.partial(objective,
                                     tracks=[track],
                                     env_steps=6_000,
                                     test_interval=3_000,
                                     test_episodes=5,
                                     action_repeat=4)
    study.optimize(objective_fn, n_trials=100, callbacks=[tensorboard_callback], catch=(ValueError,), gc_after_trial=True)

if __name__ == '__main__':
    mpo_study(track='columbia')