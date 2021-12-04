from functools import partial
from typing import Optional, Union, Dict

import tensorflow as tf

from racing.algorithms import make_mpo_agent
from racing.algorithms.d4pg import make_d4pg_agent
from racing.algorithms.lstm_mpo import make_lstm_mpo_agent
from racing.experiments.acme.experiment import SingleAgentExperiment
from racing.experiments.util import read_hyperparams

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)



def choose_agent(name: str, param_file: Union[Optional[str], Dict], checkpoint_path: str):
    if param_file:
        if isinstance(param_file, str):
            params = read_hyperparams(param_file)
        elif isinstance(param_file, Dict):
            params = param_file
        else:
            params = {}
    else:
        params ={}
    print(params.keys())
    if name == 'mpo':
        constructor = partial(make_mpo_agent, hyperparams=params, checkpoint_path=checkpoint_path)
    elif name == 'd4pg':
        constructor = partial(make_d4pg_agent, hyperparams=params, checkpoint_path=checkpoint_path)
    elif name == 'lstm-mpo':
        constructor = partial(make_lstm_mpo_agent, hyperparams=params, checkpoint_path=checkpoint_path)
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
    experiment = SingleAgentExperiment(env_config=env_config, seed=args.seed, logdir=logdir)
    agent_ctor = choose_agent(name=args.agent, param_file=args.params, checkpoint_path=checkpoint_path)
    return experiment, agent_ctor
