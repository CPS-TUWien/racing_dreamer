import copy
from typing import Dict

import acme.specs as specs
import numpy as np
import reverb
import sonnet as snt
from acme import adders, datasets
from acme.agents.agent import Agent
from acme.agents.tf import actors
from acme.agents.tf.actors import FeedForwardActor
from .agents import D4PG
from acme.agents.tf.mpo import MPO, learning
from acme.tf import networks, losses
from acme.tf.networks import StochasticModeHead
from acme.utils.loggers import Logger
from sonnet.src.optimizers.adam import Adam
from acme.tf import utils as tf2_utils
import tensorflow as tf

DEFAULT_PARAMS = dict(
    policy_layers=[400, 400, 400, 400],
    critic_layers=[400, 400, 400],
    policy_lr=1e-4,
    critic_lr=1e-4,
    sigma=0.3,
    atoms=51,
    discount=0.99,
    batch_size=256,
    target_update_period=100,
    samples_per_insert=32.0,
    n_step=5,
    clipping=True,
    checkpoint=True,
    max_replay_size=1000000,
    min_replay_size=1000,
)


def make_d4pg_agent(env_spec: specs.EnvironmentSpec, logger: Logger, checkpoint_path: str, hyperparams: Dict):
    params = DEFAULT_PARAMS.copy()
    params.update(hyperparams)
    action_size = np.prod(env_spec.actions.shape, dtype=int).item()
    policy_network = snt.Sequential([
        networks.LayerNormMLP(layer_sizes=[*params.pop('policy_layers'), action_size]),
        networks.NearZeroInitializedLinear(output_size=action_size),
        networks.TanhToSpec(env_spec.actions),
    ])

    critic_network = snt.Sequential([
        networks.CriticMultiplexer(
            critic_network=networks.LayerNormMLP(layer_sizes=[*params.pop('critic_layers'), 1])
        ),
        networks.DiscreteValuedHead(vmin=-100.0, vmax=100.0, num_atoms=params.pop('atoms'))
    ])

    observation_network = tf.identity

    # Make sure observation network is a Sonnet Module.
    observation_network = tf2_utils.to_sonnet_module(observation_network)

    actor = FeedForwardActor(policy_network=snt.Sequential([
        observation_network,
        policy_network
    ]))


    # Create optimizers.
    policy_optimizer = Adam(params.pop('policy_lr'))
    critic_optimizer = Adam(params.pop('critic_lr'))

    # The learner updates the parameters (and initializes them).
    agent = D4PG(
        environment_spec=env_spec,
        policy_network=policy_network,
        critic_network=critic_network,
        observation_network=observation_network,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        logger=logger,
        checkpoint_path=checkpoint_path,
        **params
    )
    agent.__setattr__('eval_actor', actor)
    return agent