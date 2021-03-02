from time import time

from acme.agents.agent import Agent
from dm_env import Environment


def run_episode(agent: Agent, env: Environment, action_repeat: int = 1, update: bool = False):
    start_time = time()
    episode_steps = 0
    episode_return = 0
    timestep = env.reset()
    agent.observe_first(timestep)
    while not timestep.last():
        action = agent.select_action(timestep.observation)
        for _ in range(action_repeat):
            timestep = env.step(action)
            agent.observe(action, next_timestep=timestep)
            episode_steps += 1
            episode_return += timestep.reward
            if update:
                agent.update()
            if timestep.last():
                break

    steps_per_second = episode_steps / (time() - start_time)
    result = {
        'episode_length': episode_steps,
        'episode_return': episode_return,
        'steps_per_second': steps_per_second,
    }
    return result
