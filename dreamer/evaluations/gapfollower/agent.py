import numpy as np
import tensorflow as tf
from evaluations.racing_agent import Agent
import math
from typing import Dict, Tuple
from scipy.signal import medfilt
from agents.gap_follower import GapFollower


class RacingAgent(GapFollower, Agent):

    def __init__(self):
        super().__init__(fixed_speed=False)

    def load(self, checkpoint):
        pass

    def action(self, observation: Dict[str, np.ndarray], **kwargs) -> Tuple[np.ndarray, float]:
        motor, steering = super().action(observation)
        return np.array([motor, steering]), None


if __name__ == '__main__':
    agent = RacingAgent()
    observation = np.ones(shape=(1080,))
    action = agent.action(observation)
    print()
