from functools import partial
from typing import Optional, Dict

from racing.experiments.sb3.sb_experiment import SingleAgentExperiment
from racing.experiments.util import read_hyperparams


def choose_agent(name: str, param_file: Optional[str], checkpoint_path: str):
    if param_file:
        if isinstance(param_file, str):
            params = read_hyperparams(param_file)
        elif isinstance(param_file, Dict):
            params = param_file
        else:
            params = {}
    else:
        params = {}
    if name == 'sac':
        from stable_baselines3.sac import SAC
        constructor = partial(SAC, policy='MlpPolicy', verbose=1, **params)
    elif name == 'ppo':
        from stable_baselines3.ppo import PPO
        constructor = partial(PPO, policy='MlpPolicy', verbose=1, **params)
    elif name == 'lstm-ppo':
        from stable_baselines.ppo2 import PPO2
        constructor = partial(PPO2, policy='MlpLstmPolicy', verbose=1, nminibatches=1, **params)
    else:
        raise NotImplementedError(name)
    return constructor

def make_experiment(args, logdir):
    checkpoint_path = f'{logdir}/checkpoints'
    env_config = SingleAgentExperiment.EnvConfig(
        track=args.track,
        task=args.task,
        action_repeat=args.action_repeat,
        training_time_limit=args.training_time_limit,
        eval_time_limit=args.eval_time_limit
    )
    if args.agent in ['ppo', 'sac']:
        version = 3
    elif args.agent in ['lstm-ppo']:
        version = 2
    else:
        raise NotImplementedError(args.agent)

    experiment = SingleAgentExperiment(env_config=env_config, seed=args.seed, logdir=logdir, version=version)
    agent_ctor = choose_agent(name=args.agent, param_file=args.params, checkpoint_path=checkpoint_path)
    return experiment, agent_ctor
