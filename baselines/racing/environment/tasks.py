from racecar_gym import Task, register_task
import math

class MaximizeSpeed(Task):

    def reward(self, agent_id, state, action) -> float:
        agent_state = state[agent_id]
        if agent_state['wall_collision']:
            reward = -1.0
        else:
            velocity = agent_state['velocity'][0]
            steering = math.fabs(action['steering'][0])
            reward = -math.exp(steering - velocity)
        return reward

    def done(self, agent_id, state) -> bool:
        return False

    def reset(self):
        pass

register_task(name='max_speed', task=MaximizeSpeed)