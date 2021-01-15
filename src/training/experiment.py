from random import random
from time import time
from typing import List, Tuple

import gym
from acme import wrappers, EnvironmentLoop
from acme.agents.agent import Agent
from acme.agents.tf.actors import FeedForwardActor
from acme.utils.loggers import Logger
from acme.wrappers import GymWrapper
from gym.wrappers import TimeLimit, FilterObservation
from racecar_gym import MultiAgentScenario, SingleAgentScenario
from racecar_gym.envs import ChangingTrackSingleAgentRaceEnv, ChangingTrackMultiAgentRaceEnv
from sonnet.src.nets.mlp import MLP

from src.environment import MultiAgentGymWrapper
from src.logger import TensorboardLogger
from src.wrappers import wrap_env, InfoToObservation, FixedResetMode
from src.wrappers.single_agent import ActionRepeat, Flatten, NormalizeObservations


class SingleAgentExperiment:

    def __init__(self, name: str, tracks: Tuple[List[str], List[str]], logger: Logger = None):
        self._train_tracks, self._test_tracks = tracks
        self._name = name
        self._logger = logger or TensorboardLogger(logdir=f'experiments/{name}-{time()}')
        self.train_env = self._wrap_training(self._make_env(tracks=self._train_tracks))
        self.test_env = self._wrap_test(env=self._make_env(tracks=self._test_tracks))

    def _wrap_training(self, env: gym.Env):
        env = FilterObservation(env, filter_keys=['lidar'])
        env = Flatten(env, flatten_obs=True, flatten_actions=True)
        env = NormalizeObservations(env)
        env = FixedResetMode(env, mode='random')
        env = TimeLimit(env, max_episode_steps=2000)
        env = ActionRepeat(env, n=10)
        env = GymWrapper(environment=env)
        env = wrappers.SinglePrecisionWrapper(env)
        return env

    def _wrap_test(self, env: gym.Env):
        env = FilterObservation(env, filter_keys=['lidar'])
        env = Flatten(env, flatten_obs=False, flatten_actions=True)
        env = NormalizeObservations(env)
        env = InfoToObservation(env)
        env = FixedResetMode(env, mode='grid')
        env = TimeLimit(env, max_episode_steps=2000)
        env = ActionRepeat(env, n=10)
        env = GymWrapper(environment=env)
        env = wrappers.SinglePrecisionWrapper(env)
        return env

    def _make_env(self, tracks: List[str]):
        scenarios = [SingleAgentScenario.from_spec(f'scenarios/{track}.yml', rendering=True) for track in tracks]
        env = ChangingTrackSingleAgentRaceEnv(scenarios=scenarios, order='sequential')
        return env

    def run(self, steps: int, agent: Agent, eval_every_steps: int = 1000000):
        self.test(agent)

    def test(self, agent: Agent):
        max_progress = dict([(track, 0) for track in self._test_tracks])
        dnf = dict([(track, False) for track in self._test_tracks])
        for track in self._test_tracks:
            step = self.test_env.reset()
            agent.observe_first(timestep=step)
            while not step.last():
                action = agent.select_action(observation=step.observation['lidar'])
                action = self.test_env.action_spec().generate_value()
                action = action + 2.0
                step = self.test_env.step(action=action)
                agent.observe(action=action, next_timestep=step)

                if step.observation['info_wrong_way'] and step.observation['info_progress'] > 0.9:
                    dnf[track] = True

                if not dnf[track]:
                    max_progress[track] = max(max_progress[track], step.observation['info_progress'])
                print(max_progress)
        return max_progress


if __name__ == '__main__':
    actor = FeedForwardActor(policy_network=MLP([64, 64, 2]))
    experiment = SingleAgentExperiment(name='test', tracks=(['austria', 'columbia'], ['columbia']))
    experiment.run(steps=1000, agent=actor)
    del actor