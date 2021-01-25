from acme.agents.agent import Agent
from dm_env import Environment
import numpy as np

from racing.experiments.acme.common import run_episode


def train_episode(agent: Agent, env: Environment, action_repeat: int = 1):
    result = run_episode(agent, env, action_repeat, update=True)
    return result

def train(agent: Agent, steps: int, env: Environment, action_repeat: int = 1, logger=None, current_step=0):
    step = 0
    metrics = {}
    while step < steps:
        result = train_episode(agent=agent, env=env, action_repeat=action_repeat)
        for k, v in result.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)
        step += result['episode_length']
        if logger:
            logger.write(result, step=current_step+step)
    results = dict((f'avg_{k}', np.mean(v)) for k, v in metrics.items())
    results.update(dict((f'std_{k}', np.std(v)) for k, v in metrics.items()))
    results['steps'] = step
    return results

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
