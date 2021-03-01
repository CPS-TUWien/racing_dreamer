import numpy as np
import pathlib
import tensorflow as tf
from models.racing_agent import Agent


class RacingAgent(Agent):

    def __init__(self, checkpoint_path: pathlib.Path):
        self._policy = tf.saved_model.load(checkpoint_path.as_posix())

    def action(self, observation, state=None) -> np.ndarray:
        observation = tf.expand_dims(tf.constant(observation['lidar'], dtype=tf.float32), axis=0)
        action = self._policy(observation)
        action = tf.squeeze(action)

        dict_action = {'motor': action[0].numpy(), 'steering': action[1].numpy()}
        state = None
        return dict_action, state

if __name__ == '__main__':
    agent = RacingAgent(checkpoint_path='policy')
    observation = np.ones(shape=(1080,))
    action = agent.action(observation)
    print()
