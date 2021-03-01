import numpy as np
from stable_baselines3 import PPO, SAC
from evaluations.racing_agent import Agent


class RacingAgent(Agent):

    def __init__(self, algorithm: str, checkpoint_path: str):
        if algorithm == 'ppo':
            policy = PPO.load(checkpoint_path)
        elif algorithm == 'sac':
            policy = SAC.load(checkpoint_path)
        else:
            raise NotImplementedError
        self._model = policy

    def action(self, obs, state=None, **kwargs) -> np.ndarray:
        action, _ = self._model.predict(obs['lidar'], state, deterministic=True)
        return action[0], None      # sac/ppo returns action of size (1,2)

if __name__ == '__main__':
    agent = RacingAgent(algorithm='ppo', checkpoint_path='best_model_ppo.zip')
    obs = np.ones(shape=(1080,))
    action = agent.action(obs)
    print()