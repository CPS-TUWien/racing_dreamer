import wrappers
from models.ncp.racing_ncp import RacingNCP
from models.dreamer.racing_dreamer import RacingDreamer
import pathlib

def make_env(task_name: str, action_repeat: int, time_limit: float):
    import gym
    env = gym.make(task_name)
    env = wrappers.ActionRepeat(env, action_repeat)
    env = wrappers.TimeLimit(env, int(100 * time_limit / action_repeat))
    return env

def make_agent(agent_name: str, checkpoint_dir: pathlib.Path, actor_version: str="default"):
    if agent_name == 'dreamer':
        agent = RacingDreamer(checkpoint_dir, actor_version=actor_version)
    elif agent_name == 'ncp':
        agent = RacingNCP(checkpoint_dir)
    else:
        raise NotImplementedError(f'agent {agent_name} not implemented')
    return agent