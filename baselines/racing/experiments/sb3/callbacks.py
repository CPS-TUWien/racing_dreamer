import os
from typing import Union, Optional, Dict, Any, List

import gym
import numpy as np

from racing import save_video

def make_callback(version=3, **kwargs):

    if version == 3:
        from stable_baselines3.common.callbacks import EventCallback, BaseCallback
        from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
    else:
        from stable_baselines.common.callbacks import EventCallback, BaseCallback
        from stable_baselines.common.vec_env import VecEnv, DummyVecEnv
        import tensorflow as tf

    class EvalCallback(EventCallback):
        """
        Callback for evaluating an agent.

        :param eval_env: The environment used for initialization
        :param callback_on_new_best: Callback to trigger
            when there is a new best model according to the ``mean_reward``
        :param n_eval_episodes: The number of episodes to test the agent
        :param eval_freq: Evaluate the agent every eval_freq call of the callback.
        :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
            will be saved. It will be updated at each evaluation.
        :param best_model_save_path: Path to a folder where the best model
            according to performance on the eval env will be saved.
        :param deterministic: Whether the evaluation should
            use a stochastic or deterministic actions.
        :param deterministic: Whether to render or not the environment during evaluation
        :param render: Whether to render or not the environment during evaluation
        :param verbose:
        """

        def __init__(
            self,
            best_model_path: str,
            eval_env: Union[gym.Env, VecEnv],
            tracks: List[str],
            log_path: str = None,
            action_repeat: int = 4,
            callback_on_new_best: Optional[BaseCallback] = None,
            eval_freq: int = 10000,
            render_freq: int = 50000,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
        ):
            super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
            self.action_repeat = action_repeat
            self.eval_freq = eval_freq
            self.render_freq = render_freq
            self.best_mean_reward = -np.inf
            self.last_mean_reward = -np.inf
            self.deterministic = deterministic
            self.render = render
            self.best_mean_progress = -np.inf
            self.best_model_path = best_model_path

            # Convert to VecEnv for consistency
            if not isinstance(eval_env, VecEnv):
                eval_env = DummyVecEnv([lambda: eval_env])

            if isinstance(eval_env, VecEnv):
                assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

            self.eval_env = eval_env
            self.tracks = tracks
            # Logs will be written in ``evaluations.npz``
            self.log_path = log_path
            self.evaluations_results = []
            self.evaluations_timesteps = []
            self.evaluations_length = []

        def _init_callback(self) -> None:
            if self.log_path is not None:
                os.makedirs(f'{self.log_path}/videos', exist_ok=True)

        def _run_evaluation(self, n_eval_episodes, deterministic, render, track, nth_frame=2):


            if len(self.tracks) == 1:
                max_progress_stats = dict(progress=[])
            else:
                max_progress_stats = dict([(f'progress_{track}', []) for track in self.tracks])
            frames = []
            reward_stats = []
            for i in range(n_eval_episodes):

                rewards = 0
                if len(self.tracks) == 1:
                    max_progress = dict(progress=0)
                else:
                    max_progress = dict([(f'progress_{track}', 0) for track in self.tracks])
                dnf = dict([(track, False) for track in self.tracks])
                # Avoid double reset, as VecEnv are reset automatically
                if not isinstance(self.eval_env, VecEnv) or i == 0:
                    obs = self.eval_env.reset()
                done, state = False, None
                step = 0
                while not done:
                    lidar_obs = obs['lidar']
                    action, state = self.model.predict(lidar_obs, state=state, deterministic=deterministic)
                    obs, reward, done, info = self.eval_env.step(action)
                    rewards += reward[0]
                    info = info[0]
                    if render and step % nth_frame == 0 and i == 0:
                        frames.append(self.eval_env.render(mode='birds_eye'))
                    step += 1

                    if info['wrong_way'] and info['progress'] > 0.9:
                        dnf[track] = True

                    if not dnf[track]:
                        progress = info['progress']
                        lap =  info['lap']
                        if len(self.tracks) == 1:
                            max_progress['progress'] = max(max_progress['progress'], progress + lap - 1)
                        else:
                            max_progress[f'progress_{track}'] = max(max_progress[f'progress_{track}'], progress + lap - 1)
                reward_stats.append(rewards)
                max_progress_stats['progress'].append(max_progress['progress'])
            max_progress_stats['progress_mean'] = np.mean(max_progress_stats['progress'])
            max_progress_stats['progress_std'] = np.std(max_progress_stats['progress'])
            max_progress_stats['return_mean'] = np.mean(reward_stats)
            max_progress_stats['return_std'] = np.std(reward_stats)

            if max_progress_stats['progress_mean'] > self.best_mean_progress:
                print(self.model)
                if version == 3:
                    self.model.policy.save(self.best_model_path)
                elif version == 2:
                    self.model.save(self.best_model_path)
                self.best_mean_progress = max_progress_stats['progress_mean']

            del max_progress_stats['progress']
            return max_progress_stats, frames, np.mean(reward_stats)

        def _on_step(self) -> bool:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                for track in self.tracks:
                    nth_frame = 2
                    render = True
                    max_progress, frames, rewards = self._run_evaluation(n_eval_episodes=10, deterministic=True, track=track, render=render, nth_frame=nth_frame)

                    if render and self.log_path is not None:
                        save_video(filename=f'{self.log_path}/videos/{track}-{self.n_calls * self.action_repeat}', frames=frames, fps=(100 // (nth_frame * self.action_repeat)))

                    for k in max_progress:
                        if version == 3:
                            self.logger.record(f"test/{k}", max_progress[k])
                        else:
                            summary = tf.Summary(value=[tf.Summary.Value(tag=f'test/{k}', simple_value=max_progress[k])])
                            self.locals['writer'].add_summary(summary, self.num_timesteps)
            return True

        def update_child_locals(self, locals_: Dict[str, Any]) -> None:
            """
            Update the references to the local variables.

            :param locals_: the local variables during rollout collection
            """
            if self.callback:
                self.callback.update_locals(locals_)

    return EvalCallback(**kwargs)