import sys
from functools import partial
from multiprocessing.context import Process

import tensorflow as tf

from racing.agents import make_mpo_agent
from racing.agents.d4pg import make_d4pg_agent
from racing.experiments.experiment import SingleAgentExperiment


def run_experiment(track: str, seed: int):
    experiment = SingleAgentExperiment(name=f'{track}_single_{seed}', tracks=([track], [track]), seed=seed)
    constructor = partial(make_d4pg_agent, hyperparams={})
    experiment.run(steps=2_100_000, agent_constructor=constructor, eval_every_steps=10000)

def main(args):
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    seeds = [123456789, 234567891, 345678912, 456789123, 567891234]
    runs = []
    for seed in seeds:
        run_experiment(track=args[1], seed=seed)
        #process = Process(target=run_experiment, args=('austria', seed))
        #process.start()
        #runs.append(process)
    #for process in runs:
     #   process.join()



if __name__ == '__main__':
    main(sys.argv)