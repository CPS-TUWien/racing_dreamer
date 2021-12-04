import os
import random
from dataclasses import dataclass
from time import time
from typing import List, Callable, Tuple

import gym
import imageio
import numpy as np
import tensorflow as tf
from acme.tf import savers
from acme import wrappers, make_environment_spec, Actor
from acme.agents.agent import Agent
from acme.specs import EnvironmentSpec
from acme.utils.counting import Counter
from acme.utils.loggers import Logger
from acme.wrappers import GymWrapper
from gym.wrappers import TimeLimit, FilterObservation
from racecar_gym import SingleAgentScenario
from racecar_gym.envs import ChangingTrackSingleAgentRaceEnv

from racing.environment import FixedResetMode
from racing.environment.single_agent import *
from .logger import TensorBoardLogger, PrefixedTensorBoardLogger
from racing.environment import *


def save_video(filename: str, frames, fps):
    if not os.path.exists(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))
    with imageio.get_writer(f'{filename}.mp4', fps=fps) as video:
        for frame in frames:
            video.append_data(frame)


class SingleAgentExperiment:
    @dataclass
    class EnvConfig:
        track: str
        task: str
        action_repeat: int = 4
        training_time_limit: int = 2000
        eval_time_limit: int = 4000

    def __init__(self, env_config: EnvConfig, seed: int, logdir: str = None):
        self._set_seed(seed)
        self._train_tracks, self._test_tracks = [env_config.track], [env_config.track]
        self._logdir = logdir
        self._logger = TensorBoardLogger(logdir=self._logdir)
        self.train_logger = PrefixedTensorBoardLogger(base_logger=self._logger, prefix='train')
        self.test_logger = PrefixedTensorBoardLogger(base_logger=self._logger, prefix='test')
        self._env_config = env_config
        self.train_env = self._wrap_training(self._make_env(tracks=self._train_tracks))
        self.test_env = self._wrap_test(env=self._make_env(tracks=self._test_tracks))
        self._counter = Counter()
        self._counter.increment(steps=0)



    def _set_seed(self, seed):
        tf.random.set_seed(seed)
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
        env = GymWrapper(environment=env)
        env = wrappers.SinglePrecisionWrapper(env)
        return env

    def _wrap_test(self, env: gym.Env):
        env = FilterObservation(env, filter_keys=['lidar'])
        env = Flatten(env, flatten_obs=False, flatten_actions=True)
        env = NormalizeObservations(env)
        env = InfoToObservation(env)
        env = FixedResetMode(env, mode='grid')
        env = TimeLimit(env, max_episode_steps=self._env_config.eval_time_limit)
        gym_env = ActionRepeat(env, n=self._env_config.action_repeat)
        env = GymWrapper(environment=gym_env)
        env = wrappers.SinglePrecisionWrapper(env)
        env.gym_env = gym_env
        return env

    def _make_env(self, tracks: List[str]):
        scenarios = [SingleAgentScenario.from_spec(f'scenarios/{self._env_config.task}/{track}.yml', rendering=False) for track in tracks]
        env = ChangingTrackSingleAgentRaceEnv(scenarios=scenarios, order='sequential')
        return env

    def configure_agent(self, agent_ctor):
        return agent_ctor(make_environment_spec(self.train_env), self.train_logger)

    def run(self,
            steps: int,
            agent_ctor: Callable[[EnvironmentSpec, Logger], Tuple[Agent, Actor]],
            eval_every_steps: int = 10000
            ):


        agent = self.configure_agent(agent_ctor)
        eval_actor = agent.eval_actor

        model_snapshotter = savers.Snapshotter(objects_to_save={
            'policy': eval_actor._policy_network
        }, directory=self._logdir)

        t = self._counter.get_counts()['steps']
        iterations = 0
        should_render = False
        render_interval = 5
        best_mean_progress = -np.inf
        while t < steps:
            if iterations % render_interval == 0:
                should_render = True
            else:
                should_render = False
            iterations += 1
            test_result = self.test(eval_actor, render=should_render, timestep=t)
            if test_result['progress_mean'] > best_mean_progress:
                best_mean_progress = test_result['progress_mean']
                model_snapshotter.save()

            self.test_logger.write(test_result, step=t)
            self.train(steps=eval_every_steps, agent=agent, counter=self._counter, logger=self.train_logger)
            t = self._counter.get_counts()['steps']
        self.train_env.environment.close()
        self.test_env.environment.close()

    def test(self, agent: Actor, timestep: int, render: bool = False, eval_episodes=10):
        if len(self._test_tracks) == 1:
            max_progress_stats = dict(progress=[])
        else:
            max_progress_stats = dict([(f'progress_{track}', []) for track in self._test_tracks])
        rewards = []
        for test_run in range(eval_episodes):
            if len(self._test_tracks) == 1:
                max_progress = dict(progress=0)
            else:
                max_progress = dict([(f'progress_{track}', 0) for track in self._test_tracks])
            dnf = dict([(track, False) for track in self._test_tracks])

            for track in self._test_tracks:
                frames = []
                step = self.test_env.reset()
                agent.observe_first(timestep=step)
                frames.append(self.test_env.gym_env.render(mode='birds_eye'))
                reward = 0
                while not step.last():
                    action = agent.select_action(observation=step.observation['lidar'])
                    step = self.test_env.step(action=action)
                    if render and test_run==0:
                        frames.append(self.test_env.gym_env.render(mode='birds_eye'))
                    agent.observe(action=action, next_timestep=step._replace(observation=step.observation['lidar']))
                    reward += step.reward
                    if step.observation['info_wrong_way'] and step.observation['info_progress'] > 0.9:
                        dnf[track] = True

                    if not dnf[track]:
                        progress = step.observation['info_progress']
                        lap = step.observation['info_lap']
                        if len(self._test_tracks) == 1:
                            max_progress['progress'] = max(max_progress['progress'], progress + lap - 1)
                        else:
                            max_progress[f'progress_{track}'] = max(max_progress[f'progress_{track}'], progress + lap - 1)

                if render and test_run == 0:
                    print('Save video.')
                    save_video(frames=frames, filename=f'{self._logdir}/videos/{track}-{timestep}', fps=100 // self._env_config.action_repeat)
                max_progress_stats['progress'].append(max_progress['progress'])
                rewards.append(reward)
        max_progress_stats['progress_mean'] = np.mean(max_progress_stats['progress'])
        max_progress_stats['progress_std'] = np.std(max_progress_stats['progress'])
        max_progress_stats['return_mean'] = np.mean(rewards)
        max_progress_stats['return_std'] = np.std(rewards)
        del max_progress_stats['progress']

        return max_progress_stats

    def train(self, steps: int, agent: Agent, counter: Counter, logger: TensorBoardLogger):
        t = 0
        while t < steps:
            start_time = time()
            episode_steps = 0
            episode_return = 0
            timestep = self.train_env.reset()
            agent.observe_first(timestep)
            while not timestep.last():
                action = agent.select_action(timestep.observation)
                timestep = self.train_env.step(action)
                agent.observe(action, next_timestep=timestep)
                agent.update()
                episode_steps += self._env_config.action_repeat
                episode_return += timestep.reward

            counts = counter.increment(episodes=1, steps=episode_steps)
            steps_per_second = episode_steps / (time() - start_time)
            result = {
                'episode_length': episode_steps,
                'episode_return': episode_return,
                'steps_per_second': steps_per_second,
            }
            t += episode_steps
            logger.write(result, step=counts['steps'])

    def train_step(self, agent: Agent):
        start_time = time()
        episode_steps = 0
        episode_return = 0
        timestep = self.train_env.reset()
        agent.observe_first(timestep)
        while not timestep.last():
            action = agent.select_action(timestep.observation)
            timestep = self.train_env.step(action)
            agent.observe(action, next_timestep=timestep)
            agent.update()
            episode_steps += self._env_config.action_repeat
            episode_return += timestep.reward
        steps_per_second = episode_steps / (time() - start_time)
        result = {
            'episode_length': episode_steps,
            'episode_return': episode_return,
            'steps_per_second': steps_per_second,
        }
        return result

    def run_trial(self, agent, steps):
        self.train(steps=steps, agent=agent, counter=self._counter, logger=self._logger)
        results = self.test(agent.eval_actor, timestep=0, render=False, eval_episodes=5)
        return results['progress_mean']