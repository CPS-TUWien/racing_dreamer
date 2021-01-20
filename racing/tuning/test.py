from time import time

from acme.agents.agent import Agent
from dm_env import Environment
import numpy as np

from racing.experiments.common import run_episode


def test_episode(agent: Agent, env: Environment, action_repeat: int = 1):
    result = run_episode(agent, env, action_repeat, update=False)
    return result

def test(agent: Agent, env: Environment, episodes: int, action_repeat: int = 1):
    metrics = {}
    for n in range(episodes):
        episode_result = test_episode(agent, env, action_repeat)
        for k, v in episode_result.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)
    results = dict((f'avg_{k}', np.mean(v)) for k, v in metrics.items())
    results.update(dict((f'std_{k}', np.std(v)) for k, v in metrics.items()))
    return results
