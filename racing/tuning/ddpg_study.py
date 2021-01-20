import functools
import gc
import sys
import time
from multiprocessing.context import Process
from typing import Tuple, List
import psutil
import gym
import optuna
from acme import make_environment_spec
from dm_env import Environment
from optuna import Trial
from optuna.integration import TensorBoardCallback
from racecar_gym import SingleAgentScenario, SingleAgentRaceEnv, VectorizedSingleAgentRaceEnv
from racecar_gym.envs import ChangingTrackSingleAgentRaceEnv
import numpy as np
from racing.algorithms import make_mpo_agent
from racing.logger import TensorBoardLogger, PrefixedTensorBoardLogger
from racing.tuning import util
from racing import experiments

def sample_params(trial: Trial):
    return dict(
        loss_epsilon=trial.suggest_float(name='loss_epsilon', low=0.001, high=0.1),
        loss_epsilon_mean=trial.suggest_float(name='loss_epsilon_mean', low=0.0001, high=0.01),
        loss_epsilon_stddev=trial.suggest_float(name='loss_epsilon_stddev', low=1e-7, high=1e-5),
        discount=trial.suggest_float(name='discount', low=0.95, high=1.0),
        policy_lr=trial.suggest_float(name='policy_lr', low=1e-5, high=1e-1),
        critic_lr=trial.suggest_float(name='policy_lr', low=1e-5, high=1e-1),
        checkpoint=False
    )

def setup_loggers(base_logger: TensorBoardLogger) -> Tuple[PrefixedTensorBoardLogger, PrefixedTensorBoardLogger, PrefixedTensorBoardLogger]:
    algorithm_logger = PrefixedTensorBoardLogger(base_logger=base_logger, prefix='algorithm')
    train_logger = PrefixedTensorBoardLogger(base_logger=base_logger, prefix='training')
    test_logger = PrefixedTensorBoardLogger(base_logger=base_logger, prefix='test')
    return algorithm_logger, train_logger, test_logger

def objective(trial: Trial, env_steps: int, test_interval: int, test_episodes: int, env: str, action_repeat: int = 1):

    def train(step: int):
        print(f'trial-{trial.number}: start training')
        train_env = util.wrap(env, time_limit=2000, reset_mode='random')
        summary = experiments.train(
            agent=agent,
            steps=test_interval,
            env=train_env,
            action_repeat=action_repeat,
            logger=train_logger,
            current_step=step
        )
        step += summary['steps']
        return step

    def test(step: int):
        print(f'trial-{trial.number}: start evaluation')
        test_env = util.wrap(env, time_limit=4000, reset_mode='grid')
        test_summary = experiments.test(
            agent=agent,
            env=test_env,
            episodes=test_episodes,
            action_repeat=action_repeat
        )
        test_logger.write(test_summary, step=step)
        return test_summary['avg_episode_return']

    env = gym.make(env)
    env_spec = make_environment_spec(util.wrap(env, time_limit=0, reset_mode='grid'))

    tunable_params = sample_params(trial)

    logger = TensorBoardLogger(logdir=f'hyperparams/{trial.study.study_name}/runs/run-{trial.number}')
    algorithm_logger, train_logger, test_logger = setup_loggers(base_logger=logger)

    agent = make_mpo_agent(env_spec=env_spec, logger=algorithm_logger, hyperparams=tunable_params)
    objective_value = -1e9
    step = 0

    while step < env_steps:
            print(f'RAM: {psutil.virtual_memory().used / 1024**3} ')
            step = train(step)
            objective_value = test(step)
            print(f'trial-{trial.number}: finished evaluation. logging. step: {step}')
            if np.isnan(objective_value):
                objective_value = -1e9

            trial.report(objective_value, step=step)
            if trial.should_prune():
                print(f'trial-{trial.number}: pruning.')
                env.close()
                logger.close()
                raise optuna.exceptions.TrialPruned()
    print(f'trial-{trial.number}: finished.')
    env.close()
    logger.close()
    del env
    del agent
    gc.collect()
    return objective_value


def mpo_study(study_name: str, id: int):
    print(f'Run study {study_name} on process {id}')
    tensorboard_callback = TensorBoardCallback(f"hyperparams/{study_name}/trials", metric_name="avg_return")

    study = optuna.create_study(
        study_name=study_name,
        storage='mysql+pymysql://user:password@localhost:3306/optuna',
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        sampler=optuna.samplers.TPESampler(n_startup_trials=4),
        direction='maximize',
        load_if_exists=True
    )

    objective_fn = functools.partial(objective,
                                     env='SingleAgentColumbia-v0',
                                     env_steps=300_000,
                                     test_interval=30_000,
                                     test_episodes=5,
                                     action_repeat=4
                                     )

    study.optimize(objective_fn, n_trials=100, callbacks=[tensorboard_callback], gc_after_trial=True)

if __name__ == '__main__':
    import tensorflow as tf

    name = f'mpo_hp_study'
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    mpo_study(study_name=name, id=sys.argv[0])