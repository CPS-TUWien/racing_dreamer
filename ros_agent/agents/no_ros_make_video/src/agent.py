import pathlib
from time import time, sleep
import argparse
import numpy as np
from utils import make_agent, make_env

# arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, default='SingleAgentColumbia-v0')
parser.add_argument("--agent", choices=['dreamer', 'ncp'], required=True)
parser.add_argument("--version", choices=['default', 'normalized'], required=False, default="default")
parser.add_argument("--checkpoint", type=pathlib.Path, required=True,
                    help='directory where pickle files are located')
parser.add_argument("--action_repeat", type=int, default=8, help='number of times the same action is performed')
parser.add_argument("--time_limit", type=float, default=100.0, help='max time in seconds')
parser.add_argument("--store_video", action='store_true', help='collect episode frames into a video')
parser.add_argument("--store_ncp_activity", action='store_true', help='collect network states into a file')
args = parser.parse_args()

# init env and agent
env = make_env(args.task_name, action_repeat=args.action_repeat, time_limit=args.time_limit)
agent = make_agent(args.agent, args.checkpoint, actor_version=args.version)

# set variables
init = time()
done = False
obs = env.reset(mode='grid')
agent_state = None
t = 0
returns = 0.0
skip_frames = 1
sim_video, activity_video = [], []
motors, steerings = [], []
# simulate
if args.store_ncp_activity:
    assert args.agent == 'ncp'
    import rendering
    render = rendering.RenderLTC(agent._ncp._wirings)
while not done:
    if t % 25 == 0:
        print(f"[Info] Current step: {t:}")
    action, agent_state = agent.action(obs, agent_state)
    motors.append(action['motor'])
    steerings.append(action['steering'])
    obs, rewards, done, states = env.step(action)
    sleep(0.0001)
    t += 1
    returns += rewards
    if args.store_video:
        # Currently, two rendering modes are available: 'birds_eye' and 'follow'
        image = env.render(mode='follow')
        sim_video.append(image)
    if args.store_ncp_activity and t%skip_frames==0:
        # note: the ncp state is a tuple (embed, state)
        # where `embed` is the output of conv layers, `state` is the state of each neuron of ltc cell
        embed, neurons_state = agent_state
        img = render.render(embed, neurons_state)
        activity_video.append(img)
    if t % 10 == 0:
        print(f'Step: {t}, Time: {time()-init:3f} sec')
# store video
import imageio
suffix = time()
if args.store_video:
    for format in ['mp4', 'gif']:
        writer = imageio.get_writer(f'videos/video_simulation_{args.agent}_{args.task_name}_{suffix}.{format}',
                                    fps=100 // args.action_repeat)
        for image in sim_video:
            writer.append_data(image)
        writer.close()
if args.store_ncp_activity:
    for format in ['mp4', 'gif']:
        writer = imageio.get_writer(f'videos/video_ncp_activity_{args.agent}_{args.task_name}_{suffix}.{format}',
                                    fps=100 // (args.action_repeat*skip_frames))
        for image in activity_video:
            writer.append_data(image)
        writer.close()
if args.store_video and args.store_ncp_activity:
    for format in ['mp4', 'gif']:
        writer = imageio.get_writer(f'videos/side_video_ncp_sim_and_activity_{args.agent}_{args.task_name}_{suffix}.{format}',
                                    fps=100 // (args.action_repeat*skip_frames))
        for sim_image, ncp_image in zip(sim_video, activity_video):
            image = np.concatenate([sim_image, ncp_image], axis=1)
            writer.append_data(image)
        writer.close()
# print out result
print(f"[Info] Cumulative Reward: {returns:.2f}")
print(f"[Info] Nr Sim Steps: {t * args.action_repeat}")
print(f"[Info] Simulated Time: {t * args.action_repeat / 100} seconds")
print(f"[Info] Real Time: {time() - init:.2f} seconds")
# plot action
import matplotlib.pyplot as plt
plt.plot(range(len(motors)), motors, label="motor")
plt.plot(range(len(steerings)), steerings, label="steering")
plt.show()
# close env
env.close()
