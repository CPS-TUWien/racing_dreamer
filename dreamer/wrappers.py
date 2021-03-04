import gym
import numpy as np
from PIL import Image
from scipy import ndimage
from racecar_gym.envs.multi_agent_race import MultiAgentScenario, MultiAgentRaceEnv

envs = {}


class RaceCarBaseEnv:
    def __init__(self, track, task, rendering=False):
        env_id = track
        if env_id not in envs.keys():
            scenario = MultiAgentScenario.from_spec(f"dreamer/scenarios/{task}/{track}.yml", rendering=rendering)
            envs[env_id] = MultiAgentRaceEnv(scenario=scenario)
        self._env = envs[env_id]

    def __getattr__(self, name):
        return getattr(self._env, name)


class RaceCarWrapper:
    def __init__(self, env, agent_id='A'):
        self._env = env
        self._id = agent_id  # main agent id

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def scenario(self):
        return self._env.scenario

    @property
    def agent_ids(self):
        return list(self._env.observation_space.spaces.keys())  # multi-agent ids

    @property
    def n_agents(self):
        return len(self.agent_ids)

    @property
    def observation_space(self):
        assert 'speed' not in self._env.observation_space.spaces[self._id]
        spaces = {}
        for agent_id, obss in self._env.observation_space.spaces.items():
            agent_space = {}
            for obs_name, obs_space in obss.spaces.items():
                agent_space[obs_name] = obs_space
                agent_space['speed'] = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
            spaces[agent_id] = gym.spaces.Dict(agent_space)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        flat_action_space = dict()
        for agent_id, act in self._env.action_space.spaces.items():
            flat_action_space[agent_id] = gym.spaces.Box(np.append(act['motor'].low, act['steering'].low),
                                                         np.append(act['motor'].high, act['steering'].high))
        return gym.spaces.Dict(flat_action_space)

    def step(self, actions):
        actions = {i: {'motor': actions[i][0], 'steering': actions[i][1]} for i in self.agent_ids}
        obs, reward, done, info = self._env.step(actions)
        for agent_id in self.agent_ids:
            obs[agent_id]['speed'] = np.linalg.norm(info[agent_id]['velocity'][:3])
            if 'low_res_camera' in obs[agent_id]:
                obs[agent_id]['image'] = obs[agent_id]['low_res_camera']
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        for agent_id in self.agent_ids:
            obs[agent_id]['speed'] = 0.0
            if 'low_res_camera' in obs[agent_id]:
                obs[agent_id]['image'] = obs[agent_id]['low_res_camera']
        return obs

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        self._env.close()


class FixedResetMode:
    def __init__(self, env, mode):
        self._env = env
        self._mode = mode

    def reset(self):
        return self._env.reset(mode=self._mode)

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeat:

    def __init__(self, env, amount):
        self._env = env
        self._amount = amount

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, info = None, None
        dones = {agent_id: False for agent_id in self._env.agent_ids}
        total_rewards = {agent_id: 0.0 for agent_id in self._env.agent_ids}
        current_step = 0
        while current_step < self._amount and not any(dones.values()):
            obs, rewards, dones, info = self._env.step(action)
            total_rewards = {agent_id: total_rewards[agent_id] + rewards[agent_id] for agent_id in self._env.agent_ids}
            current_step += 1
        return obs, total_rewards, dones, info


class ReduceActionSpace:

    def __init__(self, env, low, high):
        self._env = env
        self._low = np.array(low)
        self._high = np.array(high)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _normalize(self, action):
        return (action + 1) / 2 * (self._high - self._low) + self._low

    def step(self, action):
        original = {agent_id: self._normalize(action[agent_id]) for agent_id in self._env.agent_ids}
        return self._env.step(original)


class TimeLimit:

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, rewards, dones, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
            dones = {agent_id: True for agent_id in self._env.agent_ids}
            self._step = None
        return obs, rewards, dones, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class Render:

    def __init__(self, env, callbacks=None, follow_view=False):
        self._env = env
        self._callbacks = callbacks or ()
        self._follow_view = follow_view
        self._reset_videos_dict()

    def _reset_videos_dict(self):
        self._videos = {'birds_eye-A': []}  # by default: store birds-eye view on first agent
        if self._follow_view:  # optional: store follow-view from each agent
            for agent_id in self._env.agent_ids:
                self._videos[f'follow-{agent_id}'] = []

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obss, reward, dones, info = self._env.step(action)
        for k in self._videos.keys():
            mode, agent = k.split('-')
            frame = self._env.render(mode=mode, agent=agent)
            self._videos[k].append(frame)
        if any(dones.values()):
            for callback in self._callbacks:
                callback(self._videos)
        return obss, reward, dones, info

    def reset(self):
        obs = self._env.reset()
        for k in self._videos.keys():
            mode, agent = k.split('-')
            frame = self._env.render(mode=mode, agent=agent)
            self._videos[k] = [frame]
        return obs


class Collect:

    def __init__(self, env, callbacks=None, precision=32, occupancy_shape=(64, 64, 1)):
        self._env = env
        self._callbacks = callbacks or ()
        self._precision = precision
        self._episodes = [[] for _ in env.agent_ids]  # in multi-agent: store 1 episode for each agent
        self._occupancy_shape = occupancy_shape

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obss, reward, dones, info = self._env.step(action)
        obss = {agent_id: {k: self._convert(v) for k, v in obs.items()} for agent_id, obs in obss.items()}
        transition = obss.copy()
        for i, agent_id in enumerate(obss.keys()):
            transition[agent_id]['action'] = action[agent_id]
            transition[agent_id]['reward'] = reward[agent_id]
            transition[agent_id]['discount'] = info.get('discount', np.array(1 - float(dones[agent_id])))
            transition[agent_id]['progress'] = info[agent_id]['lap'] + info[agent_id]['progress'] - 1  # first lap is 1
            transition[agent_id]['time'] = info[agent_id]['time']
            self._episodes[i].append(transition[agent_id])
        if any(dones.values()):
            episodes = [{k: [t[k] for t in episode] for k in episode[0]} for episode in self._episodes]
            episodes = [{k: self._convert(v) for k, v in episode.items()} for episode in episodes]
            for callback in self._callbacks:
                callback(episodes)
        return obss, reward, dones, info

    def reset(self):
        obs = self._env.reset()
        transition = obs.copy()
        for i, agent_id in enumerate(obs.keys()):
            transition[agent_id]['action'] = np.zeros(self._env.action_space[agent_id].shape)
            transition[agent_id]['reward'] = 0.0
            transition[agent_id]['discount'] = 1.0
            transition[agent_id]['progress'] = -1.0
            transition[agent_id]['time'] = 0.0
            self._episodes[i] = [transition[agent_id]]
        return obs

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
        elif np.issubdtype(value.dtype, np.uint8):
            dtype = np.uint8
        else:
            raise NotImplementedError(value.dtype)
        return value.astype(dtype)


class NormalizeActions:

    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    @property
    def original_action_space(self):
        return gym.spaces.Box(self._low, self._high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)


class ObsDict:

    def __init__(self, env, key='obs'):
        self._env = env
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = {self._key: self._env.observation_space}
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {self._key: np.array(obs)}
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs = {self._key: np.array(obs)}
        return obs


class OneHotAction:

    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        shape = (self._env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        return space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        # if not np.allclose(reference, action):
        #  raise ValueError(f'Invalid one-hot action:\n{action}')
        return self._env.step(index)

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs:

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = self._env.observation_space.spaces
        assert 'reward' not in spaces
        spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs['reward'] = reward
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs['reward'] = 0.0
        return obs


class OccupancyMapObs:

    def __init__(self, env, neigh_size=100):
        self._env = env
        self._occupancy_map = env.scenario.world._maps['occupancy']
        self._neigh_size = neigh_size     # half-size (in pixel) of the observable sub-map centered on the agent
        self._map_size = (64, 64, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = self._env.observation_space.spaces
        assert 'lidar_occupancy' not in spaces
        spaces['lidar_occupancy'] = gym.spaces.Box(-np.inf, np.inf, shape=self._occupancy_map_size, dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    def step(self, action):

        obs, reward, done, info = self._env.step(action)
        # neigh occupancy map for reconstruction
        for agent_id in self._env.agent_ids:
            pose = info[agent_id]['pose']
            pr, pc = self._occupancy_map.to_pixel(pose)  # agent pose as center

            track_map = self._occupancy_map._map[pr - (self._neigh_size + 10):pr + (self._neigh_size + 10),
                                                 pc - (self._neigh_size + 10):pc + (self._neigh_size + 10)].copy()
            track_map = track_map.astype(np.uint8)
            track_map = ndimage.rotate(track_map, np.rad2deg(2 * np.pi - pose[-1]))  # rotate with car orientation
            cr, cc = track_map.shape[0] // 2, track_map.shape[1] // 2
            cropped = track_map[cr - self._neigh_size: cr + self._neigh_size,
                                cc - self._neigh_size: cc + self._neigh_size]
            cropped = np.array(Image.fromarray(cropped).resize(size=self._map_size[:2]))  # resize as 2d image
            cropped = np.expand_dims(cropped, axis=-1)  # add last channel
            obs[agent_id]['lidar_occupancy'] = cropped
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        for agent_id in self._env.agent_ids:
            obs[agent_id]['lidar_occupancy'] = np.zeros(self._map_size, dtype=np.uint8)
        return obs
