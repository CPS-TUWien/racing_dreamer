import copy
from typing import Dict

import acme.specs as specs
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
from acme import adders, datasets
from acme.agents.tf import actors
from acme.agents.tf.mpo import MPO, learning
from acme.tf import networks, losses
from acme.tf import utils as tf2_utils
from acme.utils.loggers import Logger
from dm_env import Environment
from reverb import Server
from sonnet.src.optimizers.adam import Adam

DEFAULT_PARAMS = dict(
    policy_layers=[400, 400, 400, 400],
    critic_layers=[400, 400, 400],
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
    batch_size=256,
    target_policy_update_period=100,
    target_critic_update_period=100,
    samples_per_insert=32.0,
    n_step=5,
    num_samples=20,
    clipping=True,
    checkpoint=True,
    max_replay_size=1000000,
    min_replay_size=1000,
)


def make_mpo_agent(env: Environment, replay_server: Server, hyperparams: Dict, logger: Logger=None):
    env_spec = specs.make_environment_spec(env)
    params = DEFAULT_PARAMS.copy()
    params.update(hyperparams)

    MAX_REPLAY_SIZE = params.pop('max_replay_size')
    MIN_REPLAY_SIZE = params.pop('min_replay_size')

    replay_table = reverb.Table(
        name='mpo_priority_table',
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=MAX_REPLAY_SIZE,
        rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1),
        signature=adders.NStepTransitionAdder.signature(env_spec)
    )

    client = reverb.Client(f'localhost:{replay_server.port}')
    client.















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

    observation_network = tf.identity

    loss_param_keys = list(filter(lambda key: key.startswith('loss_'), params.keys()))
    loss_params = dict([(k.replace('loss_', ''), params.pop(k)) for k in loss_param_keys])
    policy_loss_module = losses.MPO(**loss_params)

    # Create a replay server to add data to.

    # Make sure observation network is a Sonnet Module.
    observation_network = tf2_utils.to_sonnet_module(observation_network)

    # Create optimizers.
    policy_optimizer = Adam(params.pop('policy_lr'))
    critic_optimizer = Adam(params.pop('critic_lr'))

    # The learner updates the parameters (and initializes them).
    return MPO(
        environment_spec=env_spec,
        policy_network=policy_network,
        critic_network=critic_network,
        observation_network=observation_network,
        policy_loss_module=policy_loss_module,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        logger=logger
    )













    # The adder is used to insert observations into replay.
    address = f'localhost:{self._server.port}'
    adder = adders.NStepTransitionAdder(
        client=reverb.Client(address),
        n_step=n_step,
        discount=discount)

    # The dataset object to learn from.
    dataset = datasets.make_reverb_dataset(
        table=replay_table_name,
        server_address=address,
        batch_size=batch_size,
        prefetch_size=prefetch_size)

    # Make sure observation network is a Sonnet Module.
    observation_network = tf2_utils.to_sonnet_module(observation_network)

    # Create target networks before creating online/target network variables.
    target_policy_network = copy.deepcopy(policy_network)
    target_critic_network = copy.deepcopy(critic_network)
    target_observation_network = copy.deepcopy(observation_network)

    # Get observation and action specs.
    act_spec = environment_spec.actions
    obs_spec = environment_spec.observations
    emb_spec = tf2_utils.create_variables(observation_network, [obs_spec])

    # Create the behavior policy.
    behavior_network = snt.Sequential([
        observation_network,
        policy_network,
        networks.StochasticSamplingHead(),
    ])

    # Create variables.
    tf2_utils.create_variables(policy_network, [emb_spec])
    tf2_utils.create_variables(critic_network, [emb_spec, act_spec])
    tf2_utils.create_variables(target_policy_network, [emb_spec])
    tf2_utils.create_variables(target_critic_network, [emb_spec, act_spec])
    tf2_utils.create_variables(target_observation_network, [obs_spec])

    # Create the actor which defines how we take actions.
    actor = actors.FeedForwardActor(
        policy_network=behavior_network, adder=adder)

    # Create optimizers.
    policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)
    critic_optimizer = critic_optimizer or snt.optimizers.Adam(1e-4)

    # The learner updates the parameters (and initializes them).
    learner = learning.MPOLearner(
        policy_network=policy_network,
        critic_network=critic_network,
        observation_network=observation_network,
        target_policy_network=target_policy_network,
        target_critic_network=target_critic_network,
        target_observation_network=target_observation_network,
        policy_loss_module=policy_loss_module,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        clipping=clipping,
        discount=discount,
        num_samples=num_samples,
        target_policy_update_period=target_policy_update_period,
        target_critic_update_period=target_critic_update_period,
        dataset=dataset,
        logger=logger,
        counter=counter,
        checkpoint=checkpoint)