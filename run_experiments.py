import argparse
import os
import random
from shutil import copyfile
from time import time

from racing.experiments import dispatch_experiment


def main(args):

    timestamp = time()
    experiment_name = f'{args.track}_{args.agent}_{args.task}_{args.seed}_{timestamp}'
    logdir = f'{args.logdir}/{experiment_name}'

    if args.params is not None:
        if not os.path.exists(logdir):
            os.makedirs(logdir, exist_ok=True)
        filename = os.path.basename(args.params).split('.')[0]
        copyfile(src=args.params, dst=f'{logdir}/{filename}_{timestamp}.yml')

    experiment, agent_ctor = dispatch_experiment(args, logdir)
    experiment.run(steps=args.steps, agent_ctor=agent_ctor, eval_every_steps=args.eval_interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a single agent experiment optimizing the progress based reward.')
    parser.add_argument('--track', type=str, choices=['austria', 'columbia', 'treitlstrasse_v2'], required=True)
    parser.add_argument('--task', type=str, choices=['max_progress', 'max_speed'], required=True)
    parser.add_argument('--agent', type=str, choices=['d4pg', 'mpo', 'sac', 'ppo', 'lstm-ppo', 'lstm-mpo'], required=True)
    parser.add_argument('--seed', type=int, required=False, default=random.randint(0, 100_000_000))
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--params', type=str, required=False, help='Path to hyperparam file. If none specified, default params are loaded.')
    parser.add_argument('--training_time_limit', type=int, required=False, default=2000)
    parser.add_argument('--eval_time_limit', type=int, required=False, default=4000)
    parser.add_argument('--eval_interval', type=int, required=False, default=10_000)
    parser.add_argument('--action_repeat', type=int, required=False, default=4)
    parser.add_argument('--logdir', type=str, default='logs/experiments')
    args = parser.parse_args()
    print(args.seed)
    main(args)
