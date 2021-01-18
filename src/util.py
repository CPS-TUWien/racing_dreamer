from collections import defaultdict
from time import time
from typing import Optional, List, Tuple, Mapping, Dict

import imageio
from acme import EnvironmentLoop, core, Actor
from acme.agents.agent import Agent
from acme.utils.counting import Counter, Number
from acme.utils.loggers import TerminalLogger, Logger
from dm_env import Environment

class ActionRepeatCounter(Counter):

    def __init__(self, action_repeat: int, time_delta: float = 1.0, parent: Optional['Counter'] = None, prefix: str = '', return_only_prefixed: bool = False):
        super().__init__(parent, prefix, time_delta, return_only_prefixed)
        self._action_repeat = action_repeat

    def increment(self, **counts: Number) -> Dict[str, Number]:
        counts = dict([(k, v * self._action_repeat) for k, v in counts.items()])
        return super().increment(**counts)


def save_video(filename: str, frames, fps):
    with imageio.get_writer(filename, fps=fps) as video:
        for frame in frames:
            video.append_data(frame)


def test_actor(envs: List[Tuple[str, Environment]], actor: Actor, logger: Logger = TerminalLogger(), video = True, fps = 20):
    for name, env in envs:
        if video:
            results = run_eval_episode(env, actor, nth_frame=5)
            filename = f'logs/{name}_{time()}.mp4'
            save_video(filename=filename, frames=results['frames'], fps=fps)
        else:
            results = run_eval_episode(env, actor)

        logger.write({name: results['episode_return']})

def run_eval_episode(env: Environment, agent: Actor, nth_frame: int = None):
    timestep = env.reset()
    agent.observe_first(timestep)
    steps = 0
    rewards = defaultdict(int)
    frames = []
    while not timestep.last():
        # Generate an action from the agent's policy and step the environment.
        action = agent.select_action(timestep.observation)
        timestep = env.step(action)
        agent.observe(action, next_timestep=timestep)
        steps += 1
        for id in timestep.reward:
            rewards[id] += timestep.reward[id]
        if nth_frame is not None and steps % nth_frame == 0:
            frames.append(env.environment.render(mode='follow', agent='D'))

    result = {
        'episode_length': steps,
        'episode_return': rewards,
        'frames': frames
    }
    return result