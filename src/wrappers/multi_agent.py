from typing import Dict, Any, Collection, List, Union

import gym
from gym import Wrapper, Env, ObservationWrapper
from gym.spaces import Box
import numpy as np

class Flatten(Wrapper):

    def __init__(self, env: Env, flatten_obs=True, flatten_actions=True):
        super(Flatten, self).__init__(env)
        if flatten_obs:
            obs_space = dict((id, gym.spaces.flatten_space(env.observation_space[id])) for id in env.observation_space.spaces)
            self.observation_space = gym.spaces.Dict(spaces=obs_space)
        if flatten_actions:
            spaces = dict([
                (id, gym.spaces.flatten_space(env.action_space[id]))
                for id
                in env.action_space.spaces
            ])
            for k in spaces:
                spaces[k] = Box(low=-1.0, high=1.0, shape=spaces[k].shape)

            self.action_space = gym.spaces.Dict(spaces=spaces)


    def step(self, action: Dict):
        actions = {}
        for id in action.keys():
            actions[id] = np.clip(action[id], self.action_space[id].low, self.action_space[id].high)
            actions[id] = gym.spaces.unflatten(self.env.action_space[id], actions[id])
        obs, reward, done, info = self.env.step(actions)
        for id in obs:
            obs[id] = gym.spaces.flatten(self.env.observation_space[id], obs[id])
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for id in obs:
            obs[id] = gym.spaces.flatten(self.env.observation_space[id], obs[id])
        return obs

class NormalizeObservations(ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        spaces = {}
        self._scalers = {}
        for id in env.observation_space.spaces.keys():
            spaces[id] = gym.spaces.Box(
                low=np.zeros(env.observation_space[id].shape),
                high=np.ones(env.observation_space[id].shape),
                dtype=env.observation_space[id].dtype
            )
            self._scalers[id] = 1.0 / (spaces[id].high - spaces[id].low)
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, observation):
        obs = {}
        for id in observation.keys():
            obs[id] = (observation[id] - self.env.observation_space[id].low) * self._scalers[id]
        return obs


class ActionRepeat(Wrapper):

    def __init__(self, env, n: int):
        self._repeat = n
        super().__init__(env)

    def step(self, action):
        obs, total_reward, total_done, info = self.env.step(action)
        for _ in range(self._repeat-1):
            obs, reward, done, info = self.env.step(action)
            for id in obs.keys():
                total_reward[id] += reward[id]
                total_done[id] = done[id] or total_done[id]

        return obs, total_reward, total_done, info


class FilterObservation(ObservationWrapper):
    """Filter dictionary observations by their keys.

    Args:
        env: The environment to wrap.
        filter_keys: List of keys to be included in the observations.

    Raises:
        ValueError: If observation keys in not instance of None or
            iterable.
        ValueError: If any of the `filter_keys` are not included in
            the original `env`'s observation space

    """

    def __init__(self, env, filter_keys: Union[Collection[Any], Dict[Any, Collection[Any]]]):
        super(FilterObservation, self).__init__(env)
        self._env = env
        if isinstance(filter_keys, list):
            self._filter_keys = dict((id, filter_keys) for id in env.observation_space.spaces)
        else:
            self._filter_keys = filter_keys
        obs_space = dict()
        for id in env.observation_space.spaces:
            agent_space = env.observation_space.spaces[id]
            agent_keys = self._filter_keys[id]
            obs_space[id] = gym.spaces.Dict(spaces=dict((k, agent_space.spaces[k]) for k in agent_keys))
        self.observation_space = gym.spaces.Dict(spaces=obs_space)

    def observation(self, observation):
        filter_observation = self._filter_observation(observation)
        return filter_observation

    def _filter_observation(self, observation):
        filtered = dict()
        for id in observation.keys():
            obs = dict((k, observation[id][k]) for k in self._filter_keys[id])
            filtered[id] = obs
        return filtered