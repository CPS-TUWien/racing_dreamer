from models.dreamer.models import RSSM, RSSM_norm, ActionDecoder
import numpy as np
from typing import Dict
import tensorflow as tf
import pathlib


class RacingDreamer:

    def __init__(self, checkpoint_dir: pathlib.Path, actor_version: str):
        # action space
        self._reduced_low = np.array([0.005, -1.0])
        self._reduced_high = np.array([1.0, 1.0])
        self._max_steering_angle = 0.42
        self._max_force = 0.5
        self._max_velocity = 5.0
        # trained agent
        self._checkpoint_dir = checkpoint_dir
        self._rssm = RSSM(stoch=30, deter=200, hidden=200, act=tf.nn.elu)
        if actor_version == "default":
            self._actor = ActionDecoder(size=2, layers=4, units=400, dist="tanh_normal",
                                        init_std=5.0, act=tf.nn.elu)
        elif actor_version == "normalized":
            self._actor = ActionDecoder(size=2, layers=4, units=400, dist="normalized_tanhtransformed_normal",
                                        act=tf.nn.elu)
        else:
            raise NotImplementedError(f"actor version {actor_version} not implemented")
        # load checkpoint
        self._initialize_models()
        if (checkpoint_dir / 'rssm.pkl').exists() and (checkpoint_dir / 'actor.pkl').exists():
            self._rssm.load(checkpoint_dir / 'rssm.pkl')
            self._actor.load(checkpoint_dir / 'actor.pkl')
            print('Load treitlstrasse_ncp.')
        else:
            raise FileNotFoundError(f'checkpoint missing in {checkpoint_dir}')

    def _initialize_models(self):
        latent = self._rssm.initial(1)
        action = tf.zeros((1, 2), 'float32')
        embed = tf.zeros((1, 1080), 'float32')
        latent, _ = self._rssm.obs_step(latent, action, embed)
        feat = self._rssm.get_feat(latent)
        self._actor(feat).mode()

    def _preprocess_lidar(self, scan):
        # Step 1: clip values in simulated sensors' ranges
        min_range, max_range = 0.0, 15.0
        lidar = np.clip(scan, min_range, max_range)
        # Step 2: normalize lidar ranges in 0, 1
        lidar = (lidar - min_range) / (max_range - min_range) - 0.5
        lidar = np.expand_dims(lidar, 0).astype('float32')
        return lidar

    def postprocess_action(self, action):
        # clip value between +-1
        action = np.clip(action, -1, +1)
        # map -1 +1 to the reduced action space
        action = (action + 1) / 2 * (self._reduced_high - self._reduced_low) + self._reduced_low
        dict_action = {'motor': action[0], 'steering': action[1]}   # compatibility with racecar_gym action format
        return dict_action

    def action(self, observation: Dict[str, np.ndarray], state=None):
        scan = observation['lidar']
        embed = self._preprocess_lidar(scan)

        if state is None:
            latent = self._rssm.initial(1)
            action = np.zeros((1, 2), 'float32')
        else:
            latent, action = state
        latent, _ = self._rssm.obs_step(latent, action, embed)
        feat = self._rssm.get_feat(latent)
        action = self._actor(feat).mode()
        state = (latent, action)
        action = action[0].numpy()  # note: in state action must have 2 dims for shape matching
        action = self.postprocess_action(action)
        #print(f"[ACTION] {action}")
        return action, state

    def __call__(self, observation: Dict[str, np.ndarray]):
        return self.action(observation)
