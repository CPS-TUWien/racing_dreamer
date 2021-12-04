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
from racing.environment.single_agent import *
from racing.experiments.sb3.callbacks import make_callback


class SingleAgentExperiment:
    @dataclass
    class EnvConfig:
        track: str
        task: str
        action_repeat: int = 4
        training_time_limit: int = 2000
        eval_time_limit: int = 4000

    def __init__(self, env_config: EnvConfig, seed: int, logdir: str = '/data/logs/experiments', version=3):

        self._set_seed(seed)
        self._train_tracks, self._test_tracks = [env_config.track], [env_config.track]
        self._logdir = logdir
        self._env_config = env_config
        self.train_env = self._wrap_training(self._make_env(tracks=self._train_tracks))
        self.test_env = self._wrap_test(env=self._make_env(tracks=self._test_tracks))
        self._seed = seed
        self._version = version

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
        scenarios = [SingleAgentScenario.from_spec(f'scenarios/{self._env_config.task}/{track}.yml', rendering=False)
                     for track in tracks]
        env = ChangingTrackSingleAgentRaceEnv(scenarios=scenarios, order='sequential')
        return env

    def evaluate(self, model, env, n_eval_episodes: int, deterministic=True):

        episode_rewards, episode_lengths, max_progresses = [], [], []
        for i in range(n_eval_episodes):
            dnf = False
            max_progress = 0
            obs = env.reset()
            done, state = False, None
            episode_reward = 0.0
            episode_length = 0
            while not done:
                action, state = model.predict(obs['lidar'], state=state, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1

                if info['wrong_way'] and info['progress'] > 0.9:
                    dnf = True

                if not dnf:
                    progress = info['progress']
                    lap = info['lap']
                    max_progress = max(max_progress, progress + lap - 1)
            max_progresses.append(max_progress)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print(f'Progress: {np.mean(max_progresses)} avg, {np.std(max_progresses)} std')
        return np.mean(max_progresses)

    def run(self, steps: int, agent_ctor: Callable, eval_every_steps: int = 10000):
        eval_callback = make_callback(version=self._version,
                                      best_model_path=f'{self._logdir}/best_model',
                                      eval_env=self.test_env,
                                      tracks=self._test_tracks,
                                      action_repeat=self._env_config.action_repeat,
                                      log_path=self._logdir,
                                      eval_freq=eval_every_steps // self._env_config.action_repeat,
                                      render_freq=(10 * eval_every_steps) // self._env_config.action_repeat,
                                      deterministic=True,
                                      render=True
                                      )
        print('Logging directory: ', self._logdir)
        model = self.configure_agent(agent_ctor)
        print('start learning')
        model.learn(total_timesteps=steps, callback=[eval_callback])

    def configure_agent(self, agent_ctor):
        return agent_ctor(env=self.train_env, seed=self._seed, tensorboard_log=f'{self._logdir}')


    def run_trial(self, agent, steps):
        agent.learn(steps)
        results = self.evaluate(agent, self.test_env, n_eval_episodes=5)
        return results