import glob
import pathlib
import shutil
import numpy as np
import argparse
import tensorflow as tf
import time

from racing_agent import RacingAgent
from make_env import make_multi_track_env, wrap_wrt_track


tf.config.run_functions_eagerly(run_eagerly=True)  # we need it to resume a model without need of same batchlen

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


def copy_checkpoint(agent, checkpoint_file, outdir, checkpoint_id):
    if agent in ["dreamer", "sac", "ppo"]:
        cp_dir = outdir / f'checkpoints/{checkpoint_id}'
        cp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(checkpoint_file, cp_dir)  # copy file
    else:
        cp_dir = outdir / 'checkpoints'  # dest dir must not exist before
        cp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(checkpoint_file, cp_dir / f'{checkpoint_id}')  # copy dir


def get_checkpoint_regex(checkpoint_dir, track, agent, obs_type):
    return str(checkpoint_dir / f'{track}_{agent}_{obs_type}_[0-9]*')


def glob_checkpoints(checkpoint_dir, track, agent, obs_type="*"):
    regex = get_checkpoint_regex(checkpoint_dir, track, agent, obs_type)
    checkpoints = glob.glob(regex)
    print(f"[Info] Found {len(checkpoints)} checkpoint")
    if len(checkpoints) == 0:
        raise FileNotFoundError(f"No checkpoint matching {regex}")
    return checkpoints


def eval_agent(base_env, agent, action_repeat, basedir, writer, checkpoint_id, save_trajectories):
    # iterate over the evaluation tracks: for each run a number of eval episodes
    for track in args.tracks:
        print(f"[Info] Checkpoint {checkpoint_id + 1}, Track: {track}")
        # change track until find the current one
        while base_env.scenario.world._config.name != track:
            base_env.set_next_env()
        # wrap it to adapt logging to the current track
        env = wrap_wrt_track(base_env, action_repeat, basedir, writer, track,
                             checkpoint_id=checkpoint_id + 1, save_trajectories=save_trajectories)
        # run eval episodes
        for episode in range(args.eval_episodes):
            obs = env.reset()
            done = False
            agent_state = None
            while not done:
                obs = {id: {k: np.stack([v]) for k, v in o.items()} for id, o in
                       obs.items()}  # dream needs size (1, 1080)
                action, agent_state = agent.action(obs=obs['A'], reset=np.array([done]), state=agent_state)
                action = {'A': np.array(action)}
                obs, rewards, dones, info = env.step(action)
                done = dones['A']


def make_log_dir(args):
    out_dir = args.outdir / f'eval_{args.agent}_{args.trained_on.replace("_", "")}_{args.obs_type.replace("_", "")}_{time.time()}'
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(str(out_dir), max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    return out_dir, writer


def main(args):
    action_repeat = 8 if args.agent == "dreamer" else 4
    rendering = False
    basedir, writer = make_log_dir(args)
    base_env = make_multi_track_env(args.tracks, action_repeat=action_repeat,
                                    rendering=rendering, is_dreamer=args.agent == "dreamer")
    if args.agent == "ftg":
        # programmed methods (ftg) don't need to iterate over checkpoints
        agent = RacingAgent(args.agent, None, obs_type=args.obs_type, action_dist=args.action_dist)
        eval_agent(base_env, agent, action_repeat, basedir, writer, 0, save_trajectories=args.save_trajectories)
    else:
        # find all checkpoints in `checkpoint_dir` for the given agent` and training track
        checkpoints = glob_checkpoints(args.checkpoint_dir, args.trained_on, args.agent, args.obs_type)
        agent = RacingAgent(args.agent, checkpoints[0], obs_type=args.obs_type, action_dist=args.action_dist)
        # learned methods (dreamer, mfree) iterate over checkpoint list
        for i, checkpoint in enumerate(checkpoints):
            # load the model checkpoint and copy it to the log dir
            agent.load(checkpoint)
            copy_checkpoint(args.agent, checkpoint, basedir, checkpoint_id=i + 1)
            eval_agent(base_env, agent, action_repeat, basedir, writer, i, save_trajectories=args.save_trajectories)


def parse():
    tracks = ['austria', 'columbia', 'barcelona', 'gbr', 'treitlstrasse_v2']
    agents = ["dreamer", "d4pg", "mpo", "ppo", "sac", "ftg"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, choices=agents, required=True)
    parser.add_argument('--obs_type', type=str, choices=["lidar", "lidar_occupancy"], required=True)
    parser.add_argument('--action_dist', type=str, choices=["tanh_normal"], required=False, default="tanh_normal")
    parser.add_argument('--checkpoint_dir', type=pathlib.Path, required=False)
    parser.add_argument('--trained_on', type=str, required=False, choices=tracks, default="")
    parser.add_argument('--tracks', nargs='+', type=str, default=tracks)
    parser.add_argument('--outdir', type=pathlib.Path, required=True)
    parser.add_argument('--eval_episodes', nargs='?', type=int, default=10)
    parser.add_argument('--save_trajectories', action='store_true')
    args = parser.parse_args()
    assert args.agent == "ftg" or args.checkpoint_dir is not None
    assert args.agent == "ftg" or args.trained_on in tracks
    return args


if __name__ == "__main__":
    init = time.time()
    args = parse()
    main(args)
    print(f"\n[Info] Elapsed Time: {time.time() - init:.3f} seconds")
