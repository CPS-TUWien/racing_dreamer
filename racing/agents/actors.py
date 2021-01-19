from copy import copy
from typing import Dict, Any

import dm_env
from acme import core, types, Actor
from acme.agents.agent import Agent


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


