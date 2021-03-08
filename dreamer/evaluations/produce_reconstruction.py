import argparse
import pathlib
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import dreamer.tools
from dreamer.evaluations.make_env import make_single_track_env, make_multi_track_env, wrap_wrt_track
from dreamer.evaluations.racing_agent import RacingAgent
from dreamer.tools import lidar_to_image, preprocess

tf.config.run_functions_eagerly(run_eagerly=True)   # we need it to resume a model without need of same batchlen

for gpu in tf.config.experimental.list_physical_devices('GPU'):
  tf.config.experimental.set_memory_growth(gpu, True)

def make_log_dir(args):
  out_dir = args.outdir / f'reconstructions_dreamer_{time.time()}'
  out_dir.mkdir(parents=True, exist_ok=True)
  writer = tf.summary.create_file_writer(str(out_dir), max_queue=1000, flush_millis=20000)
  writer.set_as_default()
  return out_dir, writer

def save_dreams(basedir, agent, data, embed, image_pred, obs_type='lidar', summary_length=5, skip_frames=10):
    """ Perform dreaming and save the imagined sequences as image.
      `basedir`:  base log dir where the images will be stored
      `agent`:    instance of dreamer
      `embed`:    tensor of embeds, shape (B, E) where B is the episode length
      `data`:     dictionary of observed data, it contains observation and camera image
      `image_pred`: distribution of predicted reconstructions
      `obs_type`:   observation type, either lidar or lidar_occupancy
    """
    imagedir = basedir / "image"
    imagedir.mkdir(parents=True, exist_ok=True)
    if obs_type == 'lidar':
        truth = data['lidar'][:1] + 0.5
        recon = image_pred.mode()[:1]
        init, _ = agent._dynamics.observe(embed[:1, :summary_length],
                                          data['action'][:1, :summary_length])
        init = {k: v[:, -1] for k, v in init.items()}
        prior = agent._dynamics.imagine(data['action'][:1, summary_length:], init)
        openl = agent._decode(agent._dynamics.get_feat(prior)).mode()
        model = tf.concat([recon[:, :summary_length] + 0.5, openl + 0.5], 1)
        truth_img = lidar_to_image(truth)
        model_img = lidar_to_image(model)
    elif obs_type == 'lidar_occupancy':
        truth_img = data['lidar_occupancy'][:1]
        recon = image_pred.mode()[:1]
        recon = tf.cast(recon, tf.float32)  # concatenation requires same type
        init, _ = agent._dynamics.observe(embed[:1, :summary_length],
                                          data['action'][:1, :summary_length])
        init = {k: v[:, -1] for k, v in init.items()}
        prior = agent._dynamics.imagine(data['action'][:1, summary_length:], init)
        openl = agent._decode(agent._dynamics.get_feat(prior)).mode()
        openl = tf.cast(openl, tf.float32)
        model_img = tf.concat([recon[:, :summary_length], openl],
                              1)  # note: recon and open_l is already 0 or 1, no need scaling
    else:
        raise NotImplementedError(f"save dreams not implemented for {obs_type}")
    timestamp = time.time()
    plt.box(False)
    plt.axis(False)
    plt.ion()
    for imgs, prefix in zip([data['image'], truth_img, model_img], ["camera", "true", "recon"]):
        for ep in range(imgs.shape[0]):
            for t in range(0, imgs.shape[1], skip_frames):
                # plot black/white without borders
                plt.imshow(imgs[ep, t, :, :, :], cmap='binary')
                plt.savefig(f"{imagedir}/frame_{timestamp}_{obs_type}_{prefix}_{ep}_{t}.pdf",
                            bbox_inches='tight', transparent=True, pad_inches=0)


def dreaming(agent, cameras, lidars, occupancies, actions, obstype, basedir):
    """ Given observation, actions, camera and map image
      run 'dreaming' in latent space and store the real observation and the reconstructed one.
      """
    data = {'lidar': np.stack(np.expand_dims(lidars, 0)),
            'action': np.stack(np.expand_dims(actions, 0)),
            'lidar_occupancy': np.stack(np.expand_dims(occupancies, 0))}
    data = preprocess(data, config=None)    # note: this is ugly but since we don't use reward clipping, i can pass config None
    data['image'] = np.stack(np.expand_dims(cameras, 0))  # hack: don't preprocess image
    embed = agent._encode(data)
    post, prior = agent._dynamics.observe(embed, data['action'])
    feat = agent._dynamics.get_feat(post)
    image_pred = agent._decode(feat)
    save_dreams(basedir, agent, data, embed, image_pred, obs_type=obstype, summary_length=len(lidars) - 1)


def main(args):
    rendering = False
    basedir, writer = make_log_dir(args)
    # in order
    action_repeat = 8
    env = make_multi_track_env([args.track], action_repeat=action_repeat, rendering=rendering)
    env = wrap_wrt_track(env, action_repeat, basedir, writer, args.track, checkpoint_id=0)
    agents = []
    for checkpoint, obs_type in zip(args.checkpoints, args.obs_types):
        agent = RacingAgent("dreamer", checkpoint, obs_type=obs_type, action_dist='tanh_normal')
        agents.append(agent)
    # run eval episodes
    for i, episode in enumerate(range(args.episodes)):
        print(f"[Info] Starting episode {i+1}")
        obs = env.reset()
        done = False
        agent_state = None
        first_time = True
        cameras, lidars, occupancies, actions = [], [], [], []
        while not done:
            obs = {id: {k: np.stack([v]) for k, v in o.items()} for id, o in obs.items()}  # dream needs size (1, 1080)
            if first_time:
                for agent in agents:
                    _, _ = agent.action(obs=obs['A'], reset=np.array([done]), state=agent_state)
                first_time = False
            agent = agents[0]
            action, agent_state = agent.action(obs=obs['A'], reset=np.array([done]), state=agent_state)
            actions.append(action.numpy())
            action = {'A': action.numpy()}
            obs, rewards, dones, info = env.step(action)
            lidars.append(obs['A']['lidar'])
            occupancies.append(obs['A']['lidar_occupancy'])
            cameras.append(env.render(mode='birds_eye'))
            done = dones['A']
        # once collected an episode, for each agent
        # run the `dreaming` procedure (meaning encode, predict, decode)
        print(f"[Info] Dreaming after episode {i + 1}")
        for agent, obs_type in zip(agents, args.obs_types):
            dreaming(agent._agent, cameras, lidars, occupancies, actions, obs_type, basedir)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_types', nargs='+', type=str, choices=["lidar", "lidar_occupancy"], required=True)
    parser.add_argument('--checkpoints', nargs='+', type=pathlib.Path, required=True)
    parser.add_argument('--track', type=str, default='austria')
    parser.add_argument('--outdir', type=pathlib.Path, required=True)
    parser.add_argument('--episodes', nargs='?', type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    t0 = time.time()
    args = parse()
    main(args)
    print(f"\n[Info] Elapsed Time: {time.time() - t0:.3f} seconds")
