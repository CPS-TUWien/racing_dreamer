import collections
import functools
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import layers as tfpl

import tools as tools


class Dreamer(tools.Module):

    def __init__(self, config, datadir, actspace, obspace, writer):
        self._c = config
        self._actspace = actspace['A']
        self._obspace = obspace['A']
        self._actdim = actspace.n if hasattr(actspace, 'n') else self._actspace.shape[0]
        self._writer = writer
        self._random = np.random.RandomState(config.seed)
        with tf.device('cpu:0'):
            self._step = tf.Variable(tools.count_steps(datadir, config), dtype=tf.int64)
        self._should_pretrain = tools.Once()
        self._should_train = tools.Every(config.train_every)
        self._should_log = tools.Every(config.log_every)
        self._last_log = None
        self._last_time = time.time()
        self._metrics = collections.defaultdict(tf.metrics.Mean)
        self._metrics['expl_amount']  # Create variable for checkpoint.
        self._float = prec.global_policy().compute_dtype
        self._strategy = tf.distribute.MirroredStrategy()
        with self._strategy.scope():
            self._dataset = iter(self._strategy.experimental_distribute_dataset(
                tools.load_dataset(datadir, self._c)))
            self._build_model()

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step.numpy().item()
        tf.summary.experimental.set_step(step)
        if state is not None and reset.any():
            mask = tf.cast(1 - reset, self._float)[:, None]
            mask = tf.cast(1 - reset, self._float)[:, None]
            state = tf.nest.map_structure(lambda x: x * mask, state)
        if training:
            if self._should_train(step):  # call it only when training
                log = self._should_log(step)
                n = self._c.pretrain if self._should_pretrain() else self._c.train_steps
                print(f'[Info] Training for {n} steps.')
                with self._strategy.scope():
                    for train_step in range(n):
                        print(f'\t[Train Step] # {train_step}')
                        log_images = self._c.log_images and log and train_step == 0
                        self.train(next(self._dataset), log_images)
                if log:
                    self._write_summaries()
        action, state = self.policy(obs, state, training)
        if training:
            self._step.assign_add(len(reset) * self._c.action_repeat)
        return action, state

    @tf.function
    def policy(self, obs, state, training):
        if state is None:
            latent = self._dynamics.initial(len(obs[self._c.obs_type]))
            action = tf.zeros((len(obs[self._c.obs_type]), self._actdim), self._float)
        else:
            latent, action = state
        embed = self._encode(tools.preprocess(obs, self._c))
        latent, _ = self._dynamics.obs_step(latent, action, embed)
        feat = self._dynamics.get_feat(latent)
        if training:
            action = self._actor(feat).sample()
        else:
            action = self._actor(feat).mode()
        action = self._exploration(action, training)
        state = (latent, action)
        return action, state

    def load(self, filename):
        super().load(filename)
        self._should_pretrain()

    @tf.function()
    def train(self, data, log_images=False):
        self._strategy.experimental_run_v2(self._train, args=(data, log_images))

    def _train(self, data, log_images):
        with tf.GradientTape() as model_tape:
            embed = self._encode(data)
            post, prior = self._dynamics.observe(embed, data['action'])
            feat = self._dynamics.get_feat(post)
            image_pred = self._decode(feat)
            reward_pred = self._reward(feat)
            likes = tools.AttrDict()
            likes.image = tf.reduce_mean(image_pred.log_prob(data[self._c.obs_type]))
            likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))
            if self._c.pcont:
                pcont_pred = self._pcont(feat)
                pcont_target = self._c.discount * data['discount']
                likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
                likes.pcont *= self._c.pcont_scale
            prior_dist = self._dynamics.get_dist(prior)
            post_dist = self._dynamics.get_dist(post)
            div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            div = tf.maximum(div, self._c.free_nats)
            model_loss = self._c.kl_scale * div - sum(likes.values())
            model_loss /= float(self._strategy.num_replicas_in_sync)

        with tf.GradientTape() as actor_tape:
            imag_feat = self._imagine_ahead(post)
            reward = tf.cast(self._reward(imag_feat).mode(), 'float')  # cast: to address the output of bernoulli
            if self._c.pcont:
                pcont = self._pcont(imag_feat).mean()
            else:
                pcont = self._c.discount * tf.ones_like(reward)
            value = self._value(imag_feat).mode()
            returns = tools.lambda_return(
                reward[:-1], value[:-1], pcont[:-1],
                bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
            discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
                [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
            actor_loss = -tf.reduce_mean(discount * returns)
            actor_loss /= float(self._strategy.num_replicas_in_sync)

        with tf.GradientTape() as value_tape:
            value_pred = self._value(imag_feat)[:-1]
            target = tf.stop_gradient(returns)
            value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))
            value_loss /= float(self._strategy.num_replicas_in_sync)

        model_norm = self._model_opt(model_tape, model_loss)
        actor_norm = self._actor_opt(actor_tape, actor_loss)
        value_norm = self._value_opt(value_tape, value_loss)

        if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
            if self._c.log_scalars:
                self._scalar_summaries(
                    data, feat, prior_dist, post_dist, likes, div,
                    model_loss, value_loss, actor_loss, model_norm, value_norm,
                    actor_norm)
            if tf.equal(log_images, True):
                self._image_summaries(data, embed, image_pred)
                self._reward_summaries(data, reward_pred)

    def _build_model(self):
        acts = dict(
            elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
            leaky_relu=tf.nn.leaky_relu)
        cnn_act = acts[self._c.cnn_act]
        act = acts[self._c.dense_act]

        if self._c.obs_type == 'image':
            self._encode = ConvEncoder(32, cnn_act)
            self._decode = ConvDecoder(32, cnn_act)
        elif self._c.obs_type == 'lidar':
            self._encode = IdentityEncoder()
            self._decode = LidarDistanceDecoder(128, self._obspace['lidar'].shape)
        elif self._c.obs_type == 'lidar_occupancy':
            self._encode = IdentityEncoder()
            self._decode = LidarOccupancyDecoder()
        self._dynamics = RSSM(self._c.stoch_size, self._c.deter_size, self._c.deter_size)

        self._reward = DenseDecoder((), 2, self._c.num_units, dist=self._c.reward_out_dist, act=act)
        if self._c.pcont:
            self._pcont = DenseDecoder(
                (), 3, self._c.num_units, 'binary', act=act)
        self._value = DenseDecoder((), 3, self._c.num_units, act=act)
        self._actor = ActionDecoder(
            self._actdim, 4, self._c.num_units, self._c.action_dist,
            init_std=self._c.action_init_std, act=act)
        model_modules = [self._encode, self._dynamics, self._decode, self._reward]
        if self._c.pcont:
            model_modules.append(self._pcont)
        Optimizer = functools.partial(
            tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
            wdpattern=self._c.weight_decay_pattern)
        self._model_opt = Optimizer('model', model_modules, self._c.model_lr)
        self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
        self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
        # Do a train step to initialize all variables, including optimizer
        # statistics. Ideally, we would use batch size zero, but that doesn't work
        # in multi-GPU mode.
        self.train(next(self._dataset))

    def _exploration(self, action, training):
        if training:
            amount = self._c.expl_amount
            if self._c.expl_decay:
                amount *= 0.5 ** (tf.cast(self._step, tf.float32) / self._c.expl_decay)
            if self._c.expl_min:
                amount = tf.maximum(self._c.expl_min, amount)
            self._metrics['expl_amount'].update_state(amount)
        elif self._c.eval_noise:
            amount = self._c.eval_noise
        else:
            return tf.clip_by_value(action, -1, 1)
        if self._c.expl == 'additive_gaussian':
            return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
        if self._c.expl == 'completely_random':
            return tf.random.uniform(action.shape, -1, 1)
        if self._c.expl == 'epsilon_greedy':
            indices = tfd.Categorical(0 * action).sample()
            return tf.where(
                tf.random.uniform(action.shape[:1], 0, 1) < amount,
                tf.one_hot(indices, action.shape[-1], dtype=self._float),
                action)
        raise NotImplementedError(self._c.expl)

    def _imagine_ahead(self, post):
        if self._c.pcont:  # Last step could be terminal.
            post = {k: v[:, :-1] for k, v in post.items()}
        flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in post.items()}
        policy = lambda state: self._actor(
            tf.stop_gradient(self._dynamics.get_feat(state)), training=True).sample()
        states = tools.static_scan(
            lambda prev, _: self._dynamics.img_step(prev, policy(prev)),
            tf.range(self._c.horizon), start)
        imag_feat = self._dynamics.get_feat(states)
        return imag_feat

    def _scalar_summaries(
            self, data, feat, prior_dist, post_dist, likes, div,
            model_loss, value_loss, actor_loss, model_norm, value_norm,
            actor_norm):
        self._metrics['model_grad_norm'].update_state(model_norm)
        self._metrics['value_grad_norm'].update_state(value_norm)
        self._metrics['actor_grad_norm'].update_state(actor_norm)
        self._metrics['prior_ent'].update_state(prior_dist.entropy())
        self._metrics['post_ent'].update_state(post_dist.entropy())
        for name, logprob in likes.items():
            self._metrics[name + '_loss'].update_state(-logprob)
        self._metrics['div'].update_state(div)
        self._metrics['model_loss'].update_state(model_loss)
        self._metrics['value_loss'].update_state(value_loss)
        self._metrics['actor_loss'].update_state(actor_loss)
        self._metrics['action_ent'].update_state(self._actor(feat).entropy())

    def _image_summaries(self, data, embed, image_pred):
        summary_size = 6  # nr readme to be shown
        summary_length = 5  # nr step observed before dreaming
        if self._c.obs_type in ['image', 'lidar']:
            truth = data[self._c.obs_type][:summary_size] + 0.5
            recon = image_pred.mode()[:summary_size]
            init, _ = self._dynamics.observe(embed[:summary_size, :summary_length],
                                             data['action'][:summary_size, :summary_length])
            init = {k: v[:, -1] for k, v in init.items()}
            prior = self._dynamics.imagine(data['action'][:summary_size, summary_length:], init)
            openl = self._decode(self._dynamics.get_feat(prior)).mode()
            model = tf.concat([recon[:, :summary_length] + 0.5, openl + 0.5], 1)
            if self._c.obs_type == "lidar":
                truth = tools.lidar_to_image(truth)
                model = tools.lidar_to_image(model)
                error = model - truth
            else:
                error = (model - truth + 1) / 2
            openl = tf.concat([truth, model, error], 2)
        elif self._c.obs_type == 'lidar_occupancy':
            truth = data[self._c.obs_type][:summary_size]
            recon = image_pred.mode()[:summary_size]
            recon = tf.cast(recon, tf.float32)  # concatenation requires same type
            init, _ = self._dynamics.observe(embed[:summary_size, :summary_length],
                                             data['action'][:summary_size, :summary_length])
            init = {k: v[:, -1] for k, v in init.items()}
            prior = self._dynamics.imagine(data['action'][:summary_size, summary_length:], init)
            openl = self._decode(self._dynamics.get_feat(prior)).mode()
            openl = tf.cast(openl, tf.float32)
            model = tf.concat([recon[:, :summary_length], openl],
                              1)  # note: recon/openl is already 0 or 1, no need scaling
            error = (model - truth + 1) / 2
            openl = tf.concat([truth, model, error], 2)
        tools.graph_summary(self._writer, tools.video_summary,
                            'agent/train/autoencoder', openl, self._step, int(100 / self._c.action_repeat))

    def _reward_summaries(self, data, reward_pred):
        summary_size = 6  # nr readme to be shown
        truth = tools.reward_to_image(data['reward'][:summary_size])
        model = tools.reward_to_image(reward_pred.mode()[:summary_size])
        error = model - truth
        video_image = tf.concat([truth, model, error], 1)  # note: no T dimension, then stack over dim 1
        video_image = tf.expand_dims(video_image, axis=1)  # since no gif, expand dim=1 (T), B,H,W,C -> B,T,H,W,C
        tools.graph_summary(self._writer, tools.video_summary,
                            'agent/train/reward', video_image, self._step, int(100 / self._c.action_repeat))

    def _write_summaries(self):
        step = int(self._step.numpy())
        metrics = [(k, float(v.result())) for k, v in self._metrics.items()]
        if self._last_log is not None:
            duration = time.time() - self._last_time
            self._last_time += duration
            metrics.append(('fps', (step - self._last_log) / duration))
        self._last_log = step
        [m.reset_states() for m in self._metrics.values()]
        with (self._c.logdir / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
        [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
        print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
        self._writer.flush()



class RSSM(tools.Module):

    def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
        super().__init__()
        self._name = "rssm"
        self._activation = act
        self._stoch_size = stoch
        self._deter_size = deter
        self._hidden_size = hidden
        self._cell = tfkl.GRUCell(self._deter_size)

    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        return dict(
            mean=tf.zeros([batch_size, self._stoch_size], dtype),
            std=tf.zeros([batch_size, self._stoch_size], dtype),
            stoch=tf.zeros([batch_size, self._stoch_size], dtype),
            deter=self._cell.get_initial_state(None, batch_size, dtype))

    @tf.function
    def observe(self, embed, action, state=None):
        if state is None:
            state = self.initial(tf.shape(action)[0])
        embed = tf.transpose(embed, [1, 0, 2])
        action = tf.transpose(action, [1, 0, 2])
        post, prior = tools.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs),
            (action, embed), (state, state))
        post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()}
        prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
        return post, prior

    @tf.function
    def imagine(self, action, state=None):
        if state is None:
            state = self.initial(tf.shape(action)[0])
        assert isinstance(state, dict), state
        action = tf.transpose(action, [1, 0, 2])
        prior = tools.static_scan(self.img_step, action, state)
        prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
        return prior

    @staticmethod
    def get_feat(state):
        return tf.concat([state['stoch'], state['deter']], -1)

    @staticmethod
    def get_dist(state):
        return tfd.MultivariateNormalDiag(state['mean'], state['std'])

    @tf.function
    def obs_step(self, prev_state, prev_action, embed):
        prior = self.img_step(prev_state, prev_action)
        x = tf.concat([prior['deter'], embed], -1)
        x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)
        x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = self.get_dist({'mean': mean, 'std': std}).sample()
        post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
        return post, prior

    @tf.function
    def img_step(self, prev_state, prev_action):
        x = tf.concat([prev_state['stoch'], prev_action], -1)
        x = self.get('img1', tfkl.Dense, self._hidden_size, self._activation)(x)
        x, deter = self._cell(x, [prev_state['deter']])
        deter = deter[0]  # Keras wraps the state in a list.
        x = self.get('img2', tfkl.Dense, self._hidden_size, self._activation)(x)
        x = self.get('img3', tfkl.Dense, 2 * self._stoch_size, None)(x)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = self.get_dist({'mean': mean, 'std': std}).sample()
        prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
        return prior


class MLPLidarEncoder(tools.Module):
    def __init__(self, encoded_dim, depth, act=tf.nn.relu):
        super().__init__()
        self._name = "encoder"
        self._act = act
        self._depth = depth
        self._encoded_dim = encoded_dim

    def __call__(self, obs):
        if type(obs) == dict:
            lidar = obs['lidar']
        else:
            lidar = obs
        if len(lidar.shape) > 2:
            x = tf.reshape(lidar, shape=(-1, *lidar.shape[2:], 1))
        else:
            x = lidar
        x = self.get('flat', tfkl.Flatten)(x)
        x = self.get('dense1', tfkl.Dense, units=4 * self._depth, activation=self._act)(x)
        x = self.get('dense2', tfkl.Dense, units=2 * self._depth, activation=self._act)(x)
        x = self.get('dense3', tfkl.Dense, units=self._encoded_dim)(x)
        shape = (*lidar.shape[:-1], *x.shape[1:])
        return tf.reshape(x, shape=shape)


class IdentityEncoder(tools.Module):
    # This is a dummy encoder created for working with Lidar observations.
    # The size of the lidar scan is 1080, so we pass it directly without any compression.
    # In this way, the algorithm's structure is the same for all the observations.
    def __init__(self):
        super().__init__()
        self._name = "encoder"

    def __call__(self, obs):
        if type(obs) == dict:
            lidar = obs['lidar']
        else:
            lidar = obs
        return lidar


class LidarDistanceDecoder(tools.Module):
    def __init__(self, depth, shape, act=tf.nn.relu):
        super().__init__()
        self._name = "decoder"
        self._act = act
        self._shape = shape
        self._depth = depth

    def __call__(self, features):
        # note: features = tf.concat([state['stoch'], state['deter']], -1)])
        x = tf.reshape(features, shape=(-1, *features.shape[2:]))
        x = self.get('dense1', tfkl.Dense, units=2 * self._depth, activation=None)(x)
        x = self.get('dense2', tfkl.Dense, units=4 * self._depth, activation=self._act)(x)
        params = tfpl.IndependentNormal.params_size(self._shape[0])
        x = self.get('params', tfkl.Dense, units=params, activation=tf.nn.leaky_relu)(x)
        x = self.get('dist', tfpl.IndependentNormal, event_shape=self._shape[0])(x)
        dist = tfd.BatchReshape(x, batch_shape=features.shape[:2])
        return dist


class LidarOccupancyDecoder(tools.Module):

    def __init__(self, act=tf.nn.relu, shape=(64, 64, 1), depth=8):
        # it reconstruct the occupancy map of the surrounding area as binary img of size (64,64,1)
        super().__init__()
        self._name = "decoder"
        self._act = act
        self._depth = depth
        self._shape = shape

    def __call__(self, features):
        kwargs = dict(strides=2, activation=self._act)
        x = self.get('h1', tfkl.Dense, 8 * self._depth, None)(features)
        x = tf.reshape(x, [-1, 1, 1, 8 * self._depth])
        x = self.get('h2', tfkl.Conv2DTranspose, 4 * self._depth, 5, **kwargs)(x)
        x = self.get('h3', tfkl.Conv2DTranspose, 2 * self._depth, 5, **kwargs)(x)
        x = self.get('h4', tfkl.Conv2DTranspose, 1 * self._depth, 6, **kwargs)(x)
        x = self.get('h5', tfkl.Conv2DTranspose, 1, 6, **kwargs)(x)
        shape = tf.concat([tf.shape(features)[:-1], self._shape], axis=0)
        x = tf.reshape(x, shape)
        return tfd.Independent(tfd.Bernoulli(x), 3)  # last 3 dimensions (row, col, chan) define 1 pixel


class ConvEncoder(tools.Module):

    def __init__(self, depth=32, act=tf.nn.relu, obs_type="image"):
        super().__init__()
        self._name = "encoder"
        self._act = act
        self._depth = depth
        self._obs_type = obs_type

    def __call__(self, obs):
        kwargs = dict(strides=2, activation=self._act)
        x = obs[self._obs_type]
        x = tf.reshape(x, (-1,) + tuple(x.shape[-3:]))
        x = self.get('h1', tfkl.Conv2D, 1 * self._depth, 4, **kwargs)(x)
        x = self.get('h2', tfkl.Conv2D, 2 * self._depth, 4, **kwargs)(x)
        x = self.get('h3', tfkl.Conv2D, 4 * self._depth, 4, **kwargs)(x)
        x = self.get('h4', tfkl.Conv2D, 8 * self._depth, 4, **kwargs)(x)
        shape = tf.concat([tf.shape(obs[self._obs_type])[:-3], [32 * self._depth]], 0)
        return tf.reshape(x, shape)


class ConvDecoder(tools.Module):

    def __init__(self, depth=32, act=tf.nn.relu, shape=(64, 64, 3)):
        super().__init__()
        self._name = "decoder"
        self._act = act
        self._depth = depth
        self._shape = shape

    def __call__(self, features):
        kwargs = dict(strides=2, activation=self._act)
        x = self.get('h1', tfkl.Dense, 32 * self._depth, None)(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
        x = self.get('h2', tfkl.Conv2DTranspose, 4 * self._depth, 5, **kwargs)(x)
        x = self.get('h3', tfkl.Conv2DTranspose, 2 * self._depth, 5, **kwargs)(x)
        x = self.get('h4', tfkl.Conv2DTranspose, 1 * self._depth, 6, **kwargs)(x)
        x = self.get('h5', tfkl.Conv2DTranspose, self._shape[-1], 6, strides=2)(x)
        mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
        return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))


class DenseDecoder(tools.Module):

    def __init__(self, shape, layers, units, dist='normal', act=tf.nn.elu):
        super().__init__()
        self._name = "reward"
        self._shape = shape
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act

    def __call__(self, features):
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
        x = self.get(f'hout', tfkl.Dense, np.prod(self._shape))(x)
        x = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
        if self._dist == 'normal':
            return tfd.Independent(tfd.Normal(x, 1), len(self._shape))
        elif self._dist == 'binary':
            return tfd.Independent(tfd.Bernoulli(x), len(self._shape))
        raise NotImplementedError(self._dist)


class ActionDecoder(tools.Module):

    def __init__(self, size, layers, units, dist='tanh_normal', act=tf.nn.elu, min_std=1e-4, init_std=5, mean_scale=5):
        super().__init__()
        self._name = "actor"
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    def __call__(self, features, training=False):
        raw_init_std = np.log(np.exp(self._init_std) - 1)
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
        if self._dist == 'tanh_normal':  # Original from Dreamer
            # https://www.desmos.com/calculator/rcmcf5jwe7
            x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
            mean, std = tf.split(x, 2, -1)
            mean = self._mean_scale * tf.tanh(mean / self._mean_scale)
            std = tf.nn.softplus(std + raw_init_std) + self._min_std
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
            dist = tfd.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == 'normalized_tanhtransformed_normal':
            # Normalized variation of the original actor: (mu,std) normalized, then create tanh normal from them
            # The normalization params (moving avg, std) are updated only during training
            x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
            x = tf.reshape(x, [-1, 2 * self._size])
            x = self.get(f'hnorm', tfkl.BatchNormalization)(x, training=training)  # `training` true only in imagination
            x = tf.reshape(x, [*features.shape[:-1], -1])
            mean, std = tf.split(x, 2, -1)
            std = tf.nn.softplus(std) + self._min_std  # to have positive values
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
            dist = tfd.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        else:
            raise NotImplementedError(self._dist)
        return dist
