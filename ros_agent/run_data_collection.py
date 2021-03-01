import pathlib
import shutil

from models.dreamer.racing_dreamer import RacingDreamer
import gym
from time import sleep
import wrappers
import argparse
import time
import imageio
import numpy as np


def make_out_dir(args):
    outdir = args.outdir / f'collect_{time.time()}'
    datadir = outdir / 'episodes'
    datadir.mkdir(parents=True, exist_ok=True)
    videodir = outdir / 'videos'
    videodir.mkdir(parents=True, exist_ok=True)
    cpdir = outdir / 'checkpoint'
    shutil.copytree(args.checkpoint, cpdir)
    return datadir, videodir, cpdir


def make_env(task_name, action_repeat, time_limit):
  env = gym.make(task_name)
  env = wrappers.ActionRepeat(env, action_repeat)
  env = wrappers.TimeLimit(env, 100 * time_limit / action_repeat)
  return env


def save_video(args, video, videodir, suffix=time.time()):
  if not args.store_video:
    return
  writer = imageio.get_writer(f'{videodir}/video_{args.task_name}_{suffix}.mp4', fps=100 // args.action_repeat)
  for image in video:
    writer.append_data(image)
  writer.close()


def print_episode_stat(args, n_steps, returns, time):
  print(f'[Info] Episode: tot reward: {returns:.2f}, steps: {n_steps}, env steps: {n_steps * args.action_repeat},' \
        f'sim time: {t * args.action_repeat / 100} seconds, real time: {time:.2f} seconds')


# arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, required=True, help='e.g. SingleAgentColumbia_Gui-v0')
parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help='directory where pickle files are located')
parser.add_argument("--n_episodes", type=int, required=True, help='n episodes to collect')
parser.add_argument("--outdir", type=pathlib.Path, required=True, help='output directory')
parser.add_argument("--action_repeat", type=int, default=8, help='number of repeatition of the same action')
parser.add_argument("--time_limit", type=float, default=100.0, help='max time in seconds')
parser.add_argument("--store_video", action='store_true')
args = parser.parse_args()

datadir, videodir, cpdir = make_out_dir(args)
# init env and agent
env = make_env(args.task_name, action_repeat=args.action_repeat, time_limit=args.time_limit)
agent = RacingDreamer(args.checkpoint)
# simulate
for episode in range(args.n_episodes):
  init = time.time()
  done = False
  obs = env.reset(mode='random')
  state = None
  t = 0
  returns = 0.0
  video = []
  transitions = []
  # simulate
  while not done:
    transition = {}
    action, state = agent.action(obs, state)
    next_obs, rewards, done, states = env.step(action)
    transition['observation'] = obs
    transition['action'] = action
    transition['reward'] = rewards
    transition['next_observation'] = next_obs
    transitions.append(transition)
    obs = next_obs
    sleep(0.0001)
    t += 1
    returns += rewards
    if args.store_video:
      # Currently, two rendering modes are available: 'birds_eye' and 'follow'
      image = env.render(mode='birds_eye')
      video.append(image)
  # last frame
  image = env.render(mode='birds_eye')
  video.append(image)
  timestamp = time.time()
  save_video(args, video, videodir, suffix=timestamp)
  # print episode stat
  end = time.time()
  print_episode_stat(args, t, returns, end - init)
  # save episode to disk
  filename = datadir / f'{timestamp}-{episode}-{t}'
  np.save(filename, np.array(transitions), allow_pickle=True)
