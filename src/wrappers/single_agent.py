import gym
from gym import spaces, Wrapper, Env, ObservationWrapper
import numpy as np
from gym.spaces import Box


class ActionRepeat(Wrapper):

    def __init__(self, env, n: int):
        self._repeat = n
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        total_reward = reward
        for _ in range(self._repeat-1):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info

class Flatten(Wrapper):

    def __init__(self, env: Env, flatten_obs=True, flatten_actions=True):
        super(Flatten, self).__init__(env)
        if flatten_obs:
            self.observation_space = spaces.flatten_space(env.observation_space)
        if flatten_actions:
            self.action_space = spaces.flatten_space(env.action_space)
            self.action_space = Box(low=-1.0, high = 1.0, shape=self.action_space.shape)


    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = spaces.unflatten(self.env.action_space, action)
        obs, reward, done, info = self.env.step(action)
        return spaces.flatten(self.env.observation_space, obs), reward, done, info

    def reset(self, **kwargs):
        return spaces.flatten(self.env.observation_space, self.env.reset())


class NormalizeObservations(ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(env.observation_space.shape),
            high=np.ones(env.observation_space.shape),
            dtype=env.observation_space.dtype
        )

        self._scaler = 1.0 / (env.observation_space.high - env.observation_space.low)

    def observation(self, observation):
        return (observation - self.env.observation_space.low) * self._scaler