from copy import copy
from typing import Dict, Any, Optional, Tuple

import dm_env
from acme import core, types, Actor
from acme.agents.agent import Agent
from acme import adders
from acme import core
from acme import types
# Internal imports.
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class MultiAgentActor(Actor):

    def __init__(self, actors: Dict[Any, Actor]):
        self._actors = actors

    def select_action(self, observation: Dict) -> Dict[Any, types.NestedArray]:
        actions = dict()
        for id, actor in self._actors.items():
            action = actor.select_action(observation=observation[id])
            actions[id] = action
        return actions

    def observe_first(self, timestep: dm_env.TimeStep):
        for id in timestep.observation:
            tms = copy(timestep)
            ts = tms._replace(observation=timestep.observation[id])
            self._actors[id].observe_first(ts)

    def observe(self, action: Dict[Any, types.NestedArray], next_timestep: dm_env.TimeStep):
        for id in action.keys():
            ts = next_timestep._replace(observation=next_timestep.observation[id], reward=next_timestep.reward[id])
            self._actors[id].observe(action=action[id], next_timestep=ts)

    def update(self, **kwargs):
        for actor in self._actors.values():
            actor.update(**kwargs)

class RecurrentActor(core.Actor):
  """A recurrent actor.

  An actor based on a recurrent policy which takes non-batched observations and
  outputs non-batched actions, and keeps track of the recurrent state inside. It
  also allows adding experiences to replay and updating the weights from the
  policy on the learner.
  """

  def __init__(
      self,
      policy_network: snt.RNNCore,
      adder: Optional[adders.Adder] = None,
      variable_client: Optional[tf2_variable_utils.VariableClient] = None,
  ):
    """Initializes the actor.

    Args:
      policy_network: the (recurrent) policy to run.
      adder: the adder object to which allows to add experiences to a
        dataset/replay buffer.
      variable_client: object which allows to copy weights from the learner copy
        of the policy to the actor copy (in case they are separate).
    """
    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._network = policy_network
    self._state = None
    self._prev_state = None

  @tf.function
  def _policy(
      self,
      observation: types.NestedTensor,
      state: types.NestedTensor,
  ) -> Tuple[types.NestedTensor, types.NestedTensor]:

    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_observation = tf2_utils.add_batch_dim(observation)

    # Compute the policy, conditioned on the observation.
    policy, new_state = self._network(batched_observation, state)

    # Sample from the policy if it is stochastic.
    action = policy.sample() if isinstance(policy, tfd.Distribution) else policy

    return action, new_state

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Initialize the RNN state if necessary.
    if self._state is None:
      self._state = self._network.initial_state(1)

    # Step the recurrent policy forward given the current observation and state.
    policy_output, new_state = self._policy(observation, self._state)

    # Bookkeeping of recurrent states for the observe method.
    self._prev_state = self._state
    self._state = new_state

    # Return a numpy array with squeezed out batch dimension.
    return tf2_utils.to_numpy_squeeze(policy_output)

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

    # Set the state to None so that we re-initialize at the next policy call.
    self._state = None

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    if not self._adder:
      return
    self._adder.add(action, next_timestep)

  def update(self, wait: bool = False):
    if self._variable_client:
      self._variable_client.update(wait)

