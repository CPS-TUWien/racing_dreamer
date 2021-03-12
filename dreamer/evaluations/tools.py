import functools
import logging
import math
import pathlib
import pickle
import re

import imageio
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp
import tfplot
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class Module(tf.Module):

    def save(self, filename):
        values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
        with pathlib.Path(filename).open('wb') as f:
            pickle.dump(values, f)

    def load(self, filename):
        with pathlib.Path(filename).open('rb') as f:
            values = pickle.load(f)
        tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)

    def get(self, name, actor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, '_modules'):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = actor(*args, **kwargs)
        return self._modules[name]


def nest_summary(structure):
    if isinstance(structure, dict):
        return {k: nest_summary(v) for k, v in structure.items()}
    if isinstance(structure, list):
        return [nest_summary(v) for v in structure]
    if hasattr(structure, 'shape'):
        return str(structure.shape).replace(', ', 'x').strip('(), ')
    return '?'


def graph_summary(writer, fn, *args):
    step = tf.summary.experimental.get_step()

    def inner(*args):
        tf.summary.experimental.set_step(step)
        with writer.as_default():
            fn(*args)

    return tf.numpy_function(inner, args, [])


@tfplot.autowrap(figsize=(2, 2))
def plot_scatter(x: np.ndarray, y: np.ndarray, *, ax, min_v=-1, max_v=+1, color='red'):
    margin = .1
    ax.scatter(x, y, s=5, c=color)
    ax.set_xlim(min_v - margin, max_v + margin)
    ax.set_ylim(min_v - margin, max_v + margin)
    ax.axis('off')


@tfplot.autowrap(figsize=(2, 2))
def plot_step(x: np.ndarray, y: np.ndarray, *, ax, color='k', min_y=-1, max_y=1):
    margin = 0.1
    ax.step(x, y, color=color)
    ax.text(x[0] + margin, min_y + margin, 'return={:.2f}'.format(np.sum(y)))
    ax.set_ylim(min_y - margin, max_y + margin)


def lidar_to_image(scan, min_v=-1, max_v=+1, color: str = "k"):
    # shift pi/2 just to align for visualization
    angles = tf.linspace(math.pi / 2 - math.radians(270.0 / 2),
                         math.pi / 2 + math.radians(270.0 / 2),
                         scan.shape[-1])[::-1]
    batch_video = []
    for b in range(scan.shape[0]):
        single_episode = []
        for t in range(scan.shape[1]):
            x = scan[b, t, :] * tf.cos(angles)
            y = scan[b, t, :] * tf.sin(angles)
            data = plot_scatter(x, y, min_v=min_v, max_v=max_v, color=color)[:, :, :3]  # no alpha channel
            single_episode.append(data)
        video = tf.stack(single_episode)
        batch_video.append(video)
    return tf.stack(batch_video)


def reward_to_image(reward_data, min_y=-1, max_y=1):
    batch_video = []
    for b in range(reward_data.shape[0]):
        r = reward_data[b, :]
        x = range(r.shape[0])
        img = plot_step(x, r, min_y=min_y, max_y=max_y)[:, :, :3]  # return RGBA image, then discard "alpha" channel
        batch_video.append(img)
    return tf.stack(batch_video)


def flat_gif_summary(video, fps=10, name="lidar"):
    frames = []
    for i in range(video.shape[0]):
        frames.append(video[i].numpy().astype(np.uint8))
    imageio.mimsave('./{}.gif'.format(name), frames, fps=fps)


def video_summary(name, video, step=None, fps=100):
    name = name if isinstance(name, str) else name.decode('utf-8')
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    B, T, H, W, C = video.shape
    try:
        frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
        summary = tf1.Summary()
        image = tf1.Summary.Image(height=H * 3, width=W, colorspace=C)
        image.encoded_image_string = encode_gif(frames, fps)
        summary.value.add(tag=name + '/gif', image=image)
        tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    except (IOError, OSError) as e:
        print('GIF summaries require ffmpeg in $PATH.', e)
        frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
        tf.summary.image(name + '/grid', frames, step)


def encode_gif(frames, fps):
    from subprocess import Popen, PIPE
    h, w, c = frames[0].shape
    pxfmt = {1: 'gray', 3: 'rgb24'}[c]
    cmd = ' '.join([
        f'ffmpeg -y -f rawvideo -vcodec rawvideo',
        f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
        f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        f'-r {fps:.02f} -f gif -'])
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
    del proc
    return out


def simulate(agents, env, config, datadir, writer, prefix='train', steps=0, episodes=0, sim_state=None,
             agents_ids=None):
    if agents_ids is None:
        agents_ids = ['A']
    n_agents = len(agents_ids)
    # these are used to collect statistic of the first agent only
    cum_reward = 0.0  # episode level
    episode_progresses = []  # episode level
    max_progresses = []  # collection level
    cum_rewards = []  # collection level
    main_id = agents_ids[0]  # the agent w.r.t. we collect statistics
    # Initialize or unpack simulation state.
    if sim_state is None:
        step, episode = 0, 0
        dones = {agent_id: True for agent_id in agents_ids}
        length = np.zeros(n_agents, np.int32)
        obs = {agent_id: None for agent_id in agents_ids}
        agent_states = {agent_id: None for agent_id in agents_ids}
    else:
        step, episode, dones, length, obs, agent_states = sim_state
        cum_reward = {id: 0.0 for id in agents_ids}
    while (steps and step < steps) or (episodes and episode < episodes):
        # Reset envs if necessary.
        if any(dones.values()):
            obs = env.reset()
            if len(episode_progresses) > 0:  # at least 1 episode
                max_progresses.append(max(episode_progresses))
                cum_rewards.append(cum_reward)
            cum_reward = 0.0
        # Step agents.
        obs = {id: {k: np.stack([v]) for k, v in o.items()} for id, o in obs.items()}
        actions = dict()
        for i, agent_id in enumerate(agents_ids):
            actions[agent_id], agent_states[agent_id] = agents[i](obs[agent_id], np.stack([dones[agent_id]]),
                                                                  agent_states[agent_id])
            actions[agent_id] = np.array(actions[agent_id][0])
        assert len(actions) == len(agents_ids)
        # Step envs.
        obs, rewards, dones, infos = env.step(actions)
        # update episode-level information
        cum_reward = cum_reward + rewards[main_id]
        episode_progresses.append(infos[main_id]['lap'] + infos[main_id]['progress'] - 1)
        done = any(dones.values())
        episode += int(done)
        length += 1  # episode length until termination
        step += (int(done) * length).sum()  # num sim steps
        length *= (1 - done)
    # when the loop is over, write statistics for the 1st agent
    metrics_dict = {'progress': max_progresses,
                    'return': cum_rewards}
    summarize_collection(metrics_dict, config, datadir, writer, prefix)
    # Return new state to allow resuming the simulation.
    return (step - steps, episode - episodes, dones, length, obs, agent_states), np.mean(cum_rewards)


def summarize_collection(metrics_dict, config, datadir, writer, prefix):
    for metric_name, metric_list in metrics_dict.items():
        metrics = [(f'{prefix}/{metric_name}_mean', np.mean(metric_list)),
                   (f'{prefix}/{metric_name}_std', np.std(metric_list))]
        step = count_episodes(datadir)[1] * config.action_repeat
        with writer.as_default():  # Env might run in a different thread.
            tf.summary.experimental.set_step(step)
            [tf.summary.scalar(k, v) for k, v in metrics]


def count_videos(directory):
    filenames = directory.glob('**/*.mp4')
    return sum(1 for _ in filenames)


def count_episodes(directory):
    filenames = directory.glob('*.npz')
    lengths = [int(n.stem.rsplit('-', 1)[-1]) - 1 for n in filenames]
    episodes, steps = len(lengths), sum(lengths)
    return episodes, steps


def count_steps(datadir, config):
    return count_episodes(datadir)[1] * config.action_repeat


def load_episodes(directory, rescan, length=None, balance=False, seed=0):
    directory = pathlib.Path(directory).expanduser()
    random = np.random.RandomState(seed)
    cache = {}
    while True:
        for filename in directory.glob('*.npz'):
            if filename not in cache:
                try:
                    with filename.open('rb') as f:
                        episode = np.load(f)
                        episode = {k: episode[k] for k in episode.keys()}
                except Exception as e:
                    print(f'Could not load episode: {e}')
                    continue
                cache[filename] = episode
        keys = list(cache.keys())
        for index in random.choice(len(keys), rescan):
            episode = cache[keys[index]]
            if length:
                total = len(next(iter(episode.values())))
                available = total - length
                if available < 1:
                    print(f'[Info] Skipped short episode of length {available}.')
                    continue
                if balance:
                    index = min(random.randint(0, total), available)
                else:
                    index = int(random.randint(0, available + 1))  # +1 for include the last step in the sampled episode
                episode = {k: v[index: index + length] for k, v in episode.items()}
            yield episode


def preprocess(obs, config):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    with tf.device('cpu:0'):
        if 'image' in obs:
            obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
        if 'lidar' in obs:
            obs['lidar'] = tf.cast(obs['lidar'], dtype) / 15.0 - 0.5
        if 'lidar_occupancy' in obs:
            # note: when using `lidar_occupancy` the reconstruction models return a Bernoulli distribution
            # for this reason, we don't center the observation in 0, but let it in [0, 1]
            obs['lidar_occupancy'] = tf.cast(obs['lidar_occupancy'], dtype)
        if 'reward' in obs:
            clip_rewards = dict(none=lambda x: x, tanh=tf.tanh,
                                clip=lambda x: tf.clip_by_value(x, config.clip_rewards_min, config.clip_rewards_max))[
                config.clip_rewards]
            obs['reward'] = clip_rewards(obs['reward'])
    return obs


def load_dataset(directory, config):
    episode = next(load_episodes(directory, 1))
    types = {k: v.dtype for k, v in episode.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
    generator = lambda: load_episodes(
        directory, config.train_steps, config.batch_length,
        config.dataset_balance)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.map(functools.partial(preprocess, config=config))
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(10)
    return dataset


class SampleDist:

    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return tf.reduce_mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return tf.gather(sample, tf.argmax(logprob))[0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -tf.reduce_mean(logprob, 0)


class OneHotDist:

    def __init__(self, logits=None, probs=None):
        self._dist = tfd.Categorical(logits=logits, probs=probs)
        self._num_classes = self.mean().shape[-1]
        self._dtype = prec.global_policy().compute_dtype

    @property
    def name(self):
        return 'OneHotDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def prob(self, events):
        indices = tf.argmax(events, axis=-1)
        return self._dist.prob(indices)

    def log_prob(self, events):
        indices = tf.argmax(events, axis=-1)
        return self._dist.log_prob(indices)

    def mean(self):
        return self._dist.probs_parameter()

    def mode(self):
        return self._one_hot(self._dist.mode())

    def sample(self, amount=None):
        amount = [amount] if amount else []
        indices = self._dist.sample(*amount)
        sample = self._one_hot(indices)
        probs = self._dist.probs_parameter()
        sample += tf.cast(probs - tf.stop_gradient(probs), self._dtype)
        return sample

    def _one_hot(self, indices):
        return tf.one_hot(indices, self._num_classes, dtype=self._dtype)


class TanhBijector(tfp.bijectors.Bijector):

    def __init__(self, validate_args=False, name='tanh'):
        super().__init__(
            forward_min_event_ndims=0,
            validate_args=validate_args,
            name=name)

    def _forward(self, x):
        return tf.nn.tanh(x)

    def _inverse(self, y):
        dtype = y.dtype
        y = tf.cast(y, tf.float32)
        y = tf.where(
            tf.less_equal(tf.abs(y), 1.),
            tf.clip_by_value(y, -0.99999997, 0.99999997), y)
        y = tf.atanh(y)
        y = tf.cast(y, dtype)
        return y

    @staticmethod
    def _forward_log_det_jacobian(x):
        log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
        return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))


def lambda_return(
        reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * tf.ones_like(reward)
    dims = list(range(reward.shape.ndims))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        reward = tf.transpose(reward, dims)
        value = tf.transpose(value, dims)
        pcont = tf.transpose(pcont, dims)
    if bootstrap is None:
        bootstrap = tf.zeros_like(value[-1])
    next_values = tf.concat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns = static_scan(
        lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
        (inputs, pcont), bootstrap, reverse=True)
    if axis != 0:
        returns = tf.transpose(returns, dims)
    return returns


class Adam(tf.Module):

    def __init__(self, name, modules, lr, clip=None, wd=None, wdpattern=r'.*'):
        self._name = name
        self._modules = modules
        self._clip = clip
        self._wd = wd
        self._wdpattern = wdpattern
        self._opt = tf.optimizers.Adam(lr)
        self._opt = prec.LossScaleOptimizer(self._opt, 'dynamic')
        self._variables = None

    @property
    def variables(self):
        return self._opt.variables()

    def __call__(self, tape, loss):
        if self._variables is None:
            variables = [module.variables for module in self._modules]
            self._variables = tf.nest.flatten(variables)
            count = sum(np.prod(x.shape) for x in self._variables)
            print(f'[Init] Found {count} {self._name} parameters.')
        assert len(loss.shape) == 0, loss.shape
        with tape:
            loss = self._opt.get_scaled_loss(loss)
        grads = tape.gradient(loss, self._variables)
        grads = self._opt.get_unscaled_gradients(grads)
        norm = tf.linalg.global_norm(grads)
        if self._clip:
            grads, _ = tf.clip_by_global_norm(grads, self._clip, norm)
        if self._wd:
            context = tf.distribute.get_replica_context()
            context.merge_call(self._apply_weight_decay)
        self._opt.apply_gradients(zip(grads, self._variables))
        return norm

    def _apply_weight_decay(self, strategy):
        print('Applied weight decay to variables:')
        for var in self._variables:
            if re.search(self._wdpattern, self._name + '/' + var.name):
                print('- ' + self._name + '/' + var.name)
                strategy.extended.update(var, lambda v: self._wd * v)


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())


def args_type(default):
    if isinstance(default, bool):
        return lambda x: bool(['False', 'True'].index(x))
    if isinstance(default, int):
        return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, pathlib.Path):
        return lambda x: pathlib.Path(x).expanduser()
    return type(default)


def static_scan(fn, inputs, start, reverse=False):
    last = start
    outputs = [[] for _ in tf.nest.flatten(start)]
    indices = range(len(tf.nest.flatten(inputs)[0]))
    if reverse:
        indices = reversed(indices)
    for index in indices:
        inp = tf.nest.map_structure(lambda x: x[index], inputs)
        last = fn(last, inp)
        [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    outputs = [tf.stack(x, 0) for x in outputs]
    return tf.nest.pack_sequence_as(start, outputs)


def _mnd_sample(self, sample_shape=(), seed=None, name='sample'):
    return tf.random.normal(
        tuple(sample_shape) + tuple(self.event_shape),
        self.mean(), self.stddev(), self.dtype, seed, name)


tfd.MultivariateNormalDiag.sample = _mnd_sample


def _cat_sample(self, sample_shape=(), seed=None, name='sample'):
    assert len(sample_shape) in (0, 1), sample_shape
    assert len(self.logits_parameter().shape) == 2
    indices = tf.random.categorical(
        self.logits_parameter(), sample_shape[0] if sample_shape else 1,
        self.dtype, seed, name)
    if not sample_shape:
        indices = indices[..., 0]
    return indices


tfd.Categorical.sample = _cat_sample


class Every:

    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if self._last is None:
            self._last = step
            return True
        if step >= self._last + self._every:
            self._last += self._every
            return True
        return False


class Once:

    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False
