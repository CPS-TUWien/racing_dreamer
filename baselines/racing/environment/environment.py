import dm_env
import gym
from acme import wrappers, specs, types
from acme.wrappers import GymWrapper
from racecar_gym.envs import VectorizedMultiAgentRaceEnv, VectorizedSingleAgentRaceEnv, SingleAgentScenario, MultiAgentScenario




class MultiAgentGymWrapper(GymWrapper):

    def __init__(self, environment: gym.Env):
        super().__init__(environment)

    def step(self, action: types.NestedArray) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()

        observation, reward, done, info = self._environment.step(action)
        self._reset_next_step = all(done.values()) if isinstance(done, dict) else done == True

        if self._reset_next_step:
            truncated = info.get('TimeLimit.truncated', False)
            if truncated:
                return dm_env.truncation(reward, observation)
            return dm_env.termination(reward, observation)
        return dm_env.transition(reward, observation)

    def reward_spec(self):
        keys = self.environment.observation_space.spaces.keys()
        return dict((id, specs.Array(shape=(), dtype=float, name='reward')) for id in keys)

    def discount_spec(self):
        keys = self.environment.observation_space.spaces.keys()
        return dict((id, specs.Array(shape=(), dtype=float, name='discount')) for id in keys)


def make_single_agent_env(scenario: str, render=False):
    scenario = SingleAgentScenario.from_spec(scenario, rendering=render)
    env = VectorizedSingleAgentRaceEnv(scenarios=[scenario])
    env = wrap_env(env=env, wrapper_configs='single_agent_wrappers.yml')
    env = wrappers.GymWrapper(environment=env)
    env = wrappers.SinglePrecisionWrapper(env)
    return env

# def make_multi_agent_env(scenario: str, render=False, test=False):
#     scenario = MultiAgentScenario.from_spec(scenario, rendering=render)
#     env = VectorizedMultiAgentRaceEnv(scenarios=[scenario])
#     if test:
#         env = wrap_env(env=env, wrapper_configs='multi_agent_test_wrappers.yml')
#     else:
#         env = wrap_env(env=env, wrapper_configs='multi_agent_wrappers.yml')
#
#     env = MultiAgentGymWrapper(environment=env)
#     env = wrappers.SinglePrecisionWrapper(env)
#     return env