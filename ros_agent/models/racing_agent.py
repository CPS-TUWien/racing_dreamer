import numpy as np
from abc import abstractmethod
import pathlib

class Agent:

    @abstractmethod
    def action(self, observation, state=None) -> np.ndarray:
        pass

class RacingAgent:
    def __init__(self, algorithm: str, checkpoint_path: pathlib.Path):
        if algorithm == 'dreamer':
            from .dreamer import RacingDreamer
            actor_version="normalized"
            if str(checkpoint_path) == "/cwd/checkpoints/treitlstrasse_dreamer":
                actor_version = "default"
            self._agent = RacingDreamer(checkpoint_dir=checkpoint_path, actor_version=actor_version)
        elif algorithm == 'ncp':
            from .ncp import RacingNCP
            self._agent = RacingNCP(checkpoint_dir=checkpoint_path)
        elif algorithm in ['sac', 'ppo']:
            from .sb3 import RacingAgent as Sb3Agent
            self._agent = Sb3Agent(algorithm=algorithm, checkpoint_path=checkpoint_path)
        elif algorithm in ['mpo', 'd4pg']:
            from .acme import RacingAgent as AcmeAgent
            self._agent = AcmeAgent(checkpoint_path=checkpoint_path)
        else:
            raise NotImplementedError
        self._algorithm = algorithm

    def action(self, observation, state):
        return self._agent.action(observation, state)

