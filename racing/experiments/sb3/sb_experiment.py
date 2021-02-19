import os
import random
from dataclasses import dataclass
from typing import List, Callable

import gym
import numpy as np
from gym.wrappers import TimeLimit, FilterObservation
from racecar_gym import SingleAgentScenario
from racecar_gym.envs import ChangingTrackSingleAgentRaceEnv

from racing.environment import FixedResetMode
from racing.environment.single_agent import ActionRepeat, Flatten, NormalizeObservations
from racing.experiments.sb3.callbacks import EvalCallback


class SingleAgentExperiment:
    @dataclass
    class EnvConfig:
        track: str
        task: str
        action_repeat: int = 4
        training_time_limit: int = 2000
        eval_time_limit: int = 4000

    def __init__(self, env_config: EnvConfig, seed: int, logdir: str = '/data/logs/experiments'):
        self._set_seed(seed)
        self._train_tracks, self._test_tracks = [env_config.track], [env_config.track]
        self._logdir = logdir
        self._env_config = env_config
        self.train_env = self._wrap_training(self._make_env(tracks=self._train_tracks))
        self.test_env = self._wrap_test(env=self._make_env(tracks=self._test_tracks))
        self._seed = seed


    def _set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _wrap_training(self, env: gym.Env):
        env = FilterObservation(env, filter_keys=['lidar'])
        env = Flatten(env, flatten_obs=True, flatten_actions=True)
        env = NormalizeObservations(env)
        env = FixedResetMode(env, mode='random')
        env = TimeLimit(env, max_episode_steps=self._env_config.training_time_limit)
        env = ActionRepeat(env, n=self._env_config.action_repeat)
        return env

    def _wrap_test(self, env: gym.Env):
        env = FilterObservation(env, filter_keys=['lidar'])
        env = Flatten(env, flatten_obs=False, flatten_actions=True)
        env = NormalizeObservations(env)
        env = FixedResetMode(env, mode='grid')
        env = TimeLimit(env, max_episode_steps=self._env_config.eval_time_limit)
        env = ActionRepeat(env, n=self._env_config.action_repeat)
        return env

    def _make_env(self, tracks: List[str]):
        scenarios = [SingleAgentScenario.from_spec(f'scenarios/{self._env_config.task}/{track}.yml', rendering=False) for track in tracks]
        env = ChangingTrackSingleAgentRaceEnv(scenarios=scenarios, order='sequential')
        return env

    def run(self,  steps: int, agent_ctor: Callable, eval_every_steps: int = 10000):
        eval_callback = EvalCallback(best_model_path=f'{self._logdir}/best_model',
                                     eval_env=self.test_env,
                                     tracks=self._test_tracks,
                                     action_repeat=self._env_config.action_repeat,
                                     log_path=self._logdir,
                                     eval_freq=eval_every_steps // self._env_config.action_repeat,
                                     render_freq=(10*eval_every_steps) // self._env_config.action_repeat,
                                     deterministic=True,
                                     render=True
                                     )
        print('Logging directory: ', self._logdir)
        model = agent_ctor(env=self.train_env, seed=self._seed, tensorboard_log=f'{self._logdir}')
        model.learn(total_timesteps=steps, callback=[eval_callback])