from typing import Tuple

import numpy as np
from abc import abstractmethod


class Agent:

    @abstractmethod
    def action(self, obs, state=None) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def load(self, checkpoint):
        pass


class RacingAgent():
    def __init__(self, algorithm: str, checkpoint_path: str, **kwargs):
        if algorithm == 'dreamer':
            from evaluations.dreamer import RacingDreamer
            self._agent = RacingDreamer(checkpoint_path=checkpoint_path, **kwargs)
        elif algorithm == 'ftg':
            from evaluations.gapfollower import RacingAgent
            self._agent = RacingAgent()
        elif algorithm in ['sac', 'ppo']:
            from evaluations.sb3 import RacingAgent as Sb3Agent
            self._agent = Sb3Agent(algorithm=algorithm, checkpoint_path=checkpoint_path)
        elif algorithm in ['mpo', 'd4pg']:
            from evaluations.acme import RacingAgent as AcmeAgent
            self._agent = AcmeAgent(checkpoint_path=str(checkpoint_path))
        else:
            raise NotImplementedError

    def load(self, checkpoint):
        self._agent.load(checkpoint)

    def action(self, obs, **kwargs):
        return self._agent.action(obs, **kwargs)
