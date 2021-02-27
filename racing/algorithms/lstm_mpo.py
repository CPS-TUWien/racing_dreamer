import copy
from typing import Dict

import acme.specs as specs
import numpy as np
import sonnet as snt
from acme import adders, datasets
from acme.agents.agent import Agent
from acme.agents.tf import actors
from acme.agents.tf.actors import FeedForwardActor
from acme.tf import networks, losses
from acme.utils.loggers import Logger
from sonnet.src.optimizers.adam import Adam
from acme.tf import utils as tf2_utils

from .actors import RecurrentActor
from .agents import RecurrentMPO

import tensorflow as tf

DEFAULT_PARAMS = dict(
    policy_layers=[400, 400, 400, 400],
    critic_layers=[400, 400, 400],
    observation_layers=[200, 200],
    policy_lr=1e-4,
    critic_lr=1e-4,
    loss_epsilon=1e-1,
    loss_epsilon_penalty=1e-3,
    loss_epsilon_mean=1e-3,
    loss_epsilon_stddev=1e-6,
    loss_init_log_temperature=1.,
    loss_init_log_alpha_mean=1.,
    loss_init_log_alpha_stddev=10.,
    discount=0.99,
    batch_size=50,
    sequence_length=100,
    target_policy_update_period=100,
    target_critic_update_period=100,
    samples_per_insert=32.0,
    num_samples=20,
    clipping=True,
    checkpoint=True,
    max_replay_size=1000000
)


def make_lstm_mpo_agent(env_spec: specs.EnvironmentSpec, logger: Logger, hyperparams: Dict, checkpoint_path: str):
    params = DEFAULT_PARAMS.copy()
    params.update(hyperparams)
    action_size = np.prod(env_spec.actions.shape, dtype=int).item()
    policy_network = snt.Sequential([
        networks.LayerNormMLP(layer_sizes=[*params.pop('policy_layers'), action_size]),
        networks.MultivariateNormalDiagHead(num_dimensions=action_size)
    ])

    critic_network = snt.Sequential([
        networks.CriticMultiplexer(
            critic_network=networks.LayerNormMLP(layer_sizes=[*params.pop('critic_layers'), 1])
        )
    ])

    observation_network = networks.DeepRNN([
        networks.LayerNormMLP(layer_sizes=params.pop('observation_layers')),
        networks.LSTM(hidden_size=200)
    ])

    loss_param_keys = list(filter(lambda key: key.startswith('loss_'), params.keys()))
    loss_params = dict([(k.replace('loss_', ''), params.pop(k)) for k in loss_param_keys])
    policy_loss_module = losses.MPO(**loss_params)

    # Create a replay server to add data to.

    # Make sure observation network is a Sonnet Module.
    observation_network = tf2_utils.to_sonnet_module(observation_network)

    # Create optimizers.
    policy_optimizer = Adam(params.pop('policy_lr'))
    critic_optimizer = Adam(params.pop('critic_lr'))

    actor = RecurrentActor(networks.DeepRNN([
        observation_network,
        policy_network,
        networks.StochasticModeHead()
    ]))

    # The learner updates the parameters (and initializes them).
    return RecurrentMPO(
        environment_spec=env_spec,
        policy_network=policy_network,
        critic_network=critic_network,
        observation_network=observation_network,
        policy_loss_module=policy_loss_module,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        logger=logger,
        checkpoint_path=checkpoint_path,
        **params
    ), actor
