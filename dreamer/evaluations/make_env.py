import wrappers as wrappers
from racecar_gym.envs import MultiAgentScenario, ChangingTrackMultiAgentRaceEnv, MultiAgentRaceEnv
from callbacks import save_eval_videos, save_episodes, save_trajectory, summarize_episode, summarize_eval_episode


def make_multi_track_env(tracks, action_repeat, rendering=True, is_dreamer=True):
  # note: problem of multi-track racing env with wrapper `OccupancyMapObs` because it initializes the map once
  # ideas to solve this issue? when changing env force the update of occupancy map in wrapper?
  scenarios = [MultiAgentScenario.from_spec(f'scenarios/eval/{track}.yml', rendering=rendering) for track in tracks]
  env = ChangingTrackMultiAgentRaceEnv(scenarios=scenarios, order='manual')
  env = wrappers.RaceCarWrapper(env)
  env = wrappers.FixedResetMode(env, mode='grid')
  env = wrappers.ActionRepeat(env, action_repeat)
  if is_dreamer:
    env = wrappers.ReduceActionSpace(env, low=[0.005, -1.0], high=[1.0, 1.0])
  return env

def make_single_track_env(track, action_repeat, rendering=True):
  scenario = MultiAgentScenario.from_spec(f'scenarios/eval/{track}.yml', rendering=rendering)
  env = MultiAgentRaceEnv(scenario=scenario)
  env = wrappers.RaceCarWrapper(env)
  env = wrappers.FixedResetMode(env, mode='grid')
  env = wrappers.ActionRepeat(env, action_repeat)
  env = wrappers.ReduceActionSpace(env, low=[0.005, -1.0], high=[1.0, 1.0])
  env = wrappers.OccupancyMapObs(env)
  return env

def wrap_wrt_track(env, action_repeat, outdir, writer, track, checkpoint_id, save_trajectories=False):
  env = wrappers.OccupancyMapObs(env)
  render_callbacks = []
  render_callbacks.append(lambda videos: save_eval_videos(videos, outdir / 'videos', action_repeat, track, checkpoint_id))
  env = wrappers.Render(env, render_callbacks, follow_view=False)
  callbacks = []
  if save_trajectories:
    callbacks.append(lambda episodes: save_trajectory(episodes, outdir, action_repeat, track, checkpoint_id))
  callbacks.append(lambda episodes: summarize_eval_episode(episodes, outdir, writer, f'{track}', action_repeat))
  env = wrappers.Collect(env, callbacks)
  return env