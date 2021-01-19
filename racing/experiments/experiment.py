import os
import random
from functools import partial
from time import time
from typing import List, Tuple, Callable

import gym
import imageio
import numpy as np
import tensorflow as tf
from acme import wrappers, make_environment_spec
from acme.agents.agent import Agent
from acme.specs import EnvironmentSpec
from acme.utils.counting import Counter
from acme.utils.loggers import Logger
from acme.wrappers import GymWrapper
from gym.wrappers import TimeLimit, FilterObservation
from racecar_gym import SingleAgentScenario
from racecar_gym.envs import ChangingTrackSingleAgentRaceEnv

from racing.agents.mpo import make_mpo_agent
from racing.environment import InfoToObservation, FixedResetMode
from racing.environment.single_agent import ActionRepeat, Flatten, NormalizeObservations
from racing.logger import TensorBoardLogger, PrefixedTensorBoardLogger

ACTION_REPEAT = 4

def save_video(filename: str, frames, fps):
    if not os.path.exists(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))
    with imageio.get_writer(f'{filename}.mp4', fps=fps) as video:
        for frame in frames:
            video.append_data(frame)

class SingleAgentExperiment:

    def __init__(self, seed: int, name: str, tracks: Tuple[List[str], List[str]],  logger: Logger = None):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        self._train_tracks, self._test_tracks = tracks
        self._name = name
        self._logdir = f'experiments/{name}-{time()}'
        self._logger = logger or TensorBoardLogger(logdir=self._logdir)
        self.train_env = self._wrap_training(self._make_env(tracks=self._train_tracks))
        self.test_env = self._wrap_test(env=self._make_env(tracks=self._test_tracks))

    def _wrap_training(self, env: gym.Env):
        env = FilterObservation(env, filter_keys=['lidar'])
        env = Flatten(env, flatten_obs=True, flatten_actions=True)
        env = NormalizeObservations(env)
        env = FixedResetMode(env, mode='random')
        env = TimeLimit(env, max_episode_steps=2000)
        env = ActionRepeat(env, n=ACTION_REPEAT)
        env = GymWrapper(environment=env)
        env = wrappers.SinglePrecisionWrapper(env)
        return env

    def _wrap_test(self, env: gym.Env):
        env = FilterObservation(env, filter_keys=['lidar'])
        env = Flatten(env, flatten_obs=False, flatten_actions=True)
        env = NormalizeObservations(env)
        env = InfoToObservation(env)
        env = FixedResetMode(env, mode='grid')
        env = TimeLimit(env, max_episode_steps=4000)
        gym_env = ActionRepeat(env, n=ACTION_REPEAT)
        env = GymWrapper(environment=gym_env)
        env = wrappers.SinglePrecisionWrapper(env)
        env.gym_env = gym_env
        return env

    def _make_env(self, tracks: List[str]):
        scenarios = [SingleAgentScenario.from_spec(f'scenarios/{track}.yml', rendering=False) for track in tracks]
        env = ChangingTrackSingleAgentRaceEnv(scenarios=scenarios, order='sequential')
        return env

    def run(self, steps: int, agent_constructor: Callable[[EnvironmentSpec, Logger], Agent], eval_every_steps: int = 10000):
        train_logger = PrefixedTensorBoardLogger(base_logger=self._logger, prefix='train')
        test_logger = PrefixedTensorBoardLogger(base_logger=self._logger, prefix='test')

        env_spec = make_environment_spec(self.train_env)
        agent, eval_actor = agent_constructor(env_spec, train_logger)

        step_counter = Counter()
        t = 0
        iterations = 0
        render_interval = 5
        while t < steps:
            render = t >= render_interval * eval_every_steps * iterations
            if render:
                iterations += 1
            test_result = self.test(eval_actor, render=render, timestep=t)
            test_logger.write(test_result, step=t)
            self.train(steps=eval_every_steps, agent=agent, counter=step_counter, logger=train_logger)
            t = step_counter.get_counts()['steps']


    def test(self, agent: Agent, timestep: int, render: bool = False):
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
            while not step.last():
                action = agent.select_action(observation=step.observation['lidar'])
                step = self.test_env.step(action=action)
                if render:
                    frames.append(self.test_env.gym_env.render(mode='birds_eye'))
                agent.observe(action=action, next_timestep=step._replace(observation=step.observation['lidar']))

                if step.observation['info_wrong_way'] and step.observation['info_progress'] > 0.9:
                    dnf[track] = True

                if not dnf[track]:
                    progress = step.observation['info_progress']
                    lap = step.observation['info_lap']
                    if len(self._test_tracks) == 1:
                        max_progress['progress'] = max(max_progress['progress'], progress + lap - 1)
                    else:
                        max_progress[f'progress_{track}'] = max(max_progress[f'progress_{track}'], progress + lap - 1)

            if render:
                print('Save video.')
                save_video(frames=frames, filename=f'{self._logdir}/videos/{track}-{timestep}', fps=25)
        return max_progress

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
                episode_steps += ACTION_REPEAT
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
            episode_steps += ACTION_REPEAT
            episode_return += timestep.reward
        steps_per_second = episode_steps / (time() - start_time)
        result = {
            'episode_length': episode_steps,
            'episode_return': episode_return,
            'steps_per_second': steps_per_second,
        }
        return result


if __name__ == '__main__':
    experiment = SingleAgentExperiment(name='austria_single', tracks=(['austria'], ['austria']))
    constructor = partial(make_mpo_agent, hyperparams={})
    experiment.run(steps=5_000_000, agent_constructor=constructor, eval_every_steps=10000)
