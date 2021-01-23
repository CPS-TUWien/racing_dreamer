import argparse
import os
import random
from functools import partial
from shutil import copyfile
from time import time
from typing import Dict, Optional

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('pyyaml')

import tensorflow as tf
import yaml

from racing.algorithms import make_mpo_agent
from racing.algorithms.d4pg import make_d4pg_agent
from racing.experiments.experiment import SingleAgentExperiment


def read_hyperparams(file: str) -> Dict:
    with open(file, 'r') as f:
        return yaml.safe_load(f)

def choose_agent(name: str, param_file: Optional[str], checkpoint_path: str):
    params = read_hyperparams(param_file) if param_file else {}
    print(params.keys())
    if name == 'mpo':
        constructor = partial(make_mpo_agent, hyperparams=params, checkpoint_path=checkpoint_path)
    elif name == 'd4pg':
        constructor = partial(make_d4pg_agent, hyperparams=params, checkpoint_path=checkpoint_path)
    else:
        raise NotImplementedError(name)
    return constructor

def main(args):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    timestamp = time()
    experiment_name = f'{args.track}_{args.agent}_{args.task}_{args.seed}_{timestamp}'
    logdir = f'logs/experiments/{experiment_name}'

    if args.params is not None:
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        filename = os.path.basename(args.params).split('.')[0]
        copyfile(src=args.params, dst=f'{logdir}/{filename}_{timestamp}.yml')

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
    experiment.run(steps=args.steps, agent_ctor=agent_ctor, eval_every_steps=args.eval_interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a single agent experiment optimizing the progress based reward.')
    parser.add_argument('--track', type=str, choices=['austria', 'columbia', 'treitlstrasse'], required=True)
    parser.add_argument('--task', type=str, choices=['max_progress', 'max_speed'], required=True)
    parser.add_argument('--agent', type=str, choices=['d4pg', 'mpo'], required=True)
    parser.add_argument('--seed', type=int, required=False, default=random.randint(0, 100_000_000))
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--params', type=str, required=False, help='Path to hyperparam file. If none specified, default params are loaded.')
    parser.add_argument('--training_time_limit', type=int, required=False, default=2000)
    parser.add_argument('--eval_time_limit', type=int, required=False, default=4000)
    parser.add_argument('--eval_interval', type=int, required=False, default=10_000)
    parser.add_argument('--action_repeat', type=int, required=False, default=4)
    #parser.add_argument('--from_checkpoint', type=str, required=False)\
    args = parser.parse_args()
    print(args.seed)
    main(args)