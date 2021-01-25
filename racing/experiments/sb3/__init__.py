from functools import partial
from typing import Optional

from racing.experiments.sb3.sb_experiment import SingleAgentExperiment as Sb3Experiment
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO

from racing.experiments.util import read_hyperparams


def choose_agent(name: str, param_file: Optional[str], checkpoint_path: str):
    params = read_hyperparams(param_file) if param_file else {}
    if name == 'sac':
        constructor = partial(SAC, policy='MlpPolicy', verbose=1, **params)
    elif name == 'ppo':
        constructor = partial(PPO, policy='MlpPolicy', verbose=1, **params)
    else:
        raise NotImplementedError(name)
    return constructor

def make_experiment(args, logdir):
    checkpoint_path = f'{logdir}/checkpoints'
    env_config = Sb3Experiment.EnvConfig(
        track=args.track,
        task=args.task,
        action_repeat=args.action_repeat,
        training_time_limit=args.training_time_limit,
        eval_time_limit=args.eval_time_limit
    )
    experiment = Sb3Experiment(env_config=env_config, seed=args.seed, logdir=logdir)
    agent_ctor = choose_agent(name=args.agent, param_file=args.params, checkpoint_path=checkpoint_path)
    return experiment, agent_ctor
