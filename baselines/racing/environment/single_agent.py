import gym
from gym import spaces, Wrapper, Env, ObservationWrapper
import numpy as np
from gym.spaces import Box


class SingleAgentWrapper(Wrapper):

    def __init__(self, env, id: str = 'A'):
        super().__init__(env)
        self._id = id

    def step(self, action):
        step = super().step({self._id: action})
        step = [s[self._id] for s in step]
        return tuple(step)

    def reset(self, **kwargs):
        return super().reset(**kwargs)[self._id]

    def render(self, mode='human', **kwargs):
        return super().render(mode, agent=self._id, **kwargs)


class ActionRepeat(Wrapper):

    def __init__(self, env, n: int):
        self._repeat = n
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        total_reward = reward
        for _ in range(self._repeat - 1):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info


class Flatten(Wrapper):

    def __init__(self, env: Env, flatten_obs=True, flatten_actions=True):
        super(Flatten, self).__init__(env)
        self._flatten_obs = flatten_obs
        self._flatten_actions = flatten_actions
        if flatten_obs:
            self.observation_space = spaces.flatten_space(env.observation_space)
        if flatten_actions:
            self.action_space = spaces.flatten_space(env.action_space)
            self.action_space = Box(low=-1.0, high=1.0, shape=self.action_space.shape)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self._flatten_actions:
            action = spaces.unflatten(self.env.action_space, action)
        obs, reward, done, info = self.env.step(action)
        if self._flatten_obs:
            obs = spaces.flatten(self.env.observation_space, obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self._flatten_obs:
            return spaces.flatten(self.env.observation_space, self.env.reset(**kwargs))
        else:
            return self.env.reset(**kwargs)


class NormalizeObservations(ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if isinstance(env.observation_space, spaces.Dict):
            obs_spaces = []
            self._scaler = dict()
            for name, space in env.observation_space.spaces.items():
                obs_spaces.append(
                    (name, gym.spaces.Box(low=np.zeros(space.shape), high=np.ones(space.shape), dtype=space.dtype)))
                self._scaler[name] = 1.0 / (space.high - space.low)
            self.observation_space = spaces.Dict(obs_spaces)

        elif isinstance(env.observation_space, spaces.Box):
            self.observation_space = gym.spaces.Box(
                low=np.zeros(env.observation_space.shape),
                high=np.ones(env.observation_space.shape),
                dtype=env.observation_space.dtype
            )
            self._scaler = 1.0 / (env.observation_space.high - env.observation_space.low)

    def observation(self, observation):
        if isinstance(observation, dict):
            for name in observation:
                observation[name] = (observation[name] - self.env.observation_space.spaces[name].low) * self._scaler[
                    name]
            return observation
        else:
            return (observation - self.env.observation_space.low) * self._scaler


class ReduceActionSpace(Wrapper):

    def __init__(self, env, low, high):
        super(ReduceActionSpace, self).__init__(env)
        self._low = np.array(low)
        self._high = np.array(high)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        return self.env.step(original)
