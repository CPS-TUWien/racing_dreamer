from dm_env import Environment
from gym import Env
import gym.wrappers as gym_wrappers
import acme.wrappers as acme_wrappers
from src.wrappers import single_agent as wrappers, FixedResetMode


def _common_wrappers(env: Env):
    env = gym_wrappers.FilterObservation(env, filter_keys=['lidar'])
    env = wrappers.Flatten(env, flatten_obs=True, flatten_actions=True)
    env = wrappers.NormalizeObservations(env)
    return env

def wrap(env: Env, time_limit: int, reset_mode: str) -> Environment:
    env = _common_wrappers(env)
    env = FixedResetMode(env, mode=reset_mode)
    env = gym_wrappers.TimeLimit(env, max_episode_steps=time_limit)
    env = acme_wrappers.GymWrapper(environment=env)
    env = acme_wrappers.SinglePrecisionWrapper(env)
    return env

