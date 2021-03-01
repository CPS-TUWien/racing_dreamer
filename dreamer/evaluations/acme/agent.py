import numpy as np
import tensorflow as tf
from evaluations.racing_agent import Agent


class RacingAgent(Agent):

    def __init__(self, checkpoint_path):
        self.load(checkpoint_path)

    def action(self, obs, state=None, **kwargs) -> np.ndarray:
        observation = tf.constant(obs['lidar'], dtype=tf.float32)
        action = self._policy(observation)
        action = tf.squeeze(action)
        return action.numpy(), None     # second var is the state (state-less in this case)

    def load(self, checkpoint_path):
        self._policy = tf.saved_model.load(str(checkpoint_path))


if __name__ == '__main__':
    agent = RacingAgent(checkpoint_path='policy')
    observation = np.ones(shape=(1080,))
    action = agent.action(observation)
    print()
