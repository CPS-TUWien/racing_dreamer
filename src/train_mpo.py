import copy
from time import sleep, time

import imageio
import reverb
from acme import environment_loop
from acme.agents.agent import Agent
from acme.agents.tf import mpo
from acme.agents.tf.actors import FeedForwardActor
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.adders import reverb as adders
from acme.utils import loggers
from acme.utils.loggers import Dispatcher, TerminalLogger
from dm_env import Environment

from src.actors import MultiAgentActor
from src.agents import MPO
from src.environment import make_single_agent_env, make_multi_agent_env
import acme.specs as specs
import sonnet as snt
import numpy as np

from src.logger import TensorboardLogger
import src.util as util

env = make_multi_agent_env(scenario='scenarios/columbia.yml', render=True)
test_envs = [(track, make_multi_agent_env(scenario=f'scenarios/{track}.yml', render=False, test=True)) for track in ['columbia']]
env_spec = specs.make_environment_spec(env)
action_size = np.prod(env_spec.actions['A'].shape, dtype=int)
obs_spec, action_spec = env_spec.observations['A'], env_spec.actions['A']

policy_network = snt.Sequential([
    networks.LayerNormMLP(layer_sizes=[400, 400, action_size]),
    networks.MultivariateNormalDiagHead(num_dimensions=action_size)
])

critic_network = snt.Sequential([
    networks.CriticMultiplexer(
        critic_network=networks.LayerNormMLP(layer_sizes=[400, 400, 1])
    )
])

target_policy_network = copy.deepcopy(policy_network)
target_critic_network = copy.deepcopy(critic_network)

tf2_utils.create_variables(policy_network, input_spec=[obs_spec])
tf2_utils.create_variables(target_policy_network, input_spec=[obs_spec])
tf2_utils.create_variables(critic_network, input_spec=[obs_spec, action_spec])
tf2_utils.create_variables(target_critic_network, input_spec=[obs_spec, action_spec])


terminal_logger = TerminalLogger()
mpo_logger = TensorboardLogger(logdir=f'logs/mpo-{time()}', prefix='MPO')


actors = dict()
for id in env.observation_spec():
    adder = adders.NStepTransitionAdder(
        client=reverb.Client(address),
        n_step=n_step,
        discount=discount
    )

    actor = FeedForwardActor(policy_network=behavior_network, adder=adder)
    actors[id] = actor
actor = MultiAgentActor(actors)

agent = MPO(
    environment_spec=env_spec,
    policy_network=policy_network,
    critic_network=critic_network,
    logger=mpo_logger,
)
test_actors = dict([(id, agent) for id in env.observation_spec().keys()])
train_loop = environment_loop.EnvironmentLoop(env, agent, logger=mpo_logger)
for epoch in range(1000):
    util.test_actor(envs=test_envs, actor=agent, logger=mpo_logger, fps=20)
    train_loop.run(num_episodes=100)
util.test_actor(envs=test_envs, actor=agent, logger=mpo_logger)
