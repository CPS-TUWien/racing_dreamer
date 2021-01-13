from racecar_gym import Task
import math

class NStepProgress(Task):

    def __init__(self, n_steps: int):
        self._n_steps = n_steps
        self._reference_progress = 0.0
        self._current_step = 0

    def reward(self, agent_id, state, action) -> float:
        agent_state = state[agent_id]
        progress = agent_state['progress']
        reward = 0.0
        if self._current_step == 0:
            self._reference_progress = progress
        elif self._current_step == self._n_steps:
            reward = agent_state['velocity'][0] #abs(progress - self._reference_progress)
            self._reference_progress = progress #* agent_state['velocity'][0]
            self._current_step = 0
        if agent_state['wall_collision']:
            reward = -1.0
        else:
            reward = -math.exp(-agent_state['velocity'][0] + math.fabs(action['steering'][0]))

        self._current_step += 1
        return reward


    def done(self, agent_id, state) -> bool:
        return False

    def reset(self):
        self._reference_progress = 0.0
        self._current_step = 0
