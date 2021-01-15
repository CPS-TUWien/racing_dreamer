from typing import Dict

import acme.specs as specs
import numpy as np
import sonnet as snt
from acme.agents.tf.mpo import MPO
from acme.tf import networks, losses
from acme.utils.loggers import Logger
from sonnet.src.optimizers.adam import Adam


def make_mpo_agent(env_spec: specs.EnvironmentSpec, hyperparams: Dict, logger: Logger):
    action_size = np.prod(env_spec.actions.shape, dtype=int).item()
    policy_network = snt.Sequential([
        networks.LayerNormMLP(layer_sizes=[*hyperparams.pop('policy_layers'), action_size]),
        networks.MultivariateNormalDiagHead(num_dimensions=action_size)
    ])

    critic_network = snt.Sequential([
        networks.CriticMultiplexer(
            critic_network=networks.LayerNormMLP(layer_sizes=[*hyperparams.pop('critic_layers'), 1])
        )
    ])

    agent = MPO(
        environment_spec=env_spec,
        policy_network=policy_network,
        critic_network=critic_network,
        logger=logger,
        policy_loss_module=losses.MPO(**hyperparams.pop('loss_params')),
        policy_optimizer=Adam(hyperparams.pop('policy_lr')),
        critic_optimizer=Adam(hyperparams.pop('critic_lr')),
        **hyperparams
    )
    return agent
