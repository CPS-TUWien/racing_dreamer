# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MPO agent implementation."""

import copy
from typing import NamedTuple

from acme import datasets
from acme import specs
from acme import types
from acme.adders import reverb as adders
from acme.agents import agent
from acme.agents.tf.actors import FeedForwardActor
from acme.agents.tf.mpo import learning
from acme.specs import EnvironmentSpec
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import reverb
import sonnet as snt
import tensorflow as tf

from src.actors import MultiAgentActor


def make_mpo(env_spec: EnvironmentSpec, multi_agent: bool = False):

    single_env_spec = specs.EnvironmentSpec(
        observations=environment_spec.observations['A'],
        actions=environment_spec.actions['A'],
        rewards=environment_spec.rewards['A'],
        discounts=environment_spec.discounts['A']
    )
    # Create a replay server to add data to.
    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1),
        signature=adders.NStepTransitionAdder.signature(single_env_spec))

    self._server = reverb.Server([replay_table], port=None)

    # The adder is used to insert observations into replay.
    address = f'localhost:{self._server.port}'


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
    act_spec = single_env_spec.actions
    obs_spec = single_env_spec.observations
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
    actors = dict()
    for id in environment_spec.observations:
        adder = adders.NStepTransitionAdder(
            client=reverb.Client(address),
            n_step=n_step,
            discount=discount
        )

        actor = FeedForwardActor(policy_network=behavior_network, adder=adder)
        actors[id] = actor
    actor = MultiAgentActor(actors)

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

    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(batch_size, min_replay_size),
        observations_per_step=float(batch_size) / samples_per_insert)
