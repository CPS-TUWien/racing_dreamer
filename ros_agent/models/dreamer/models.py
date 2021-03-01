import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import layers as tfpl


import tools as tools


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

    def get_feat(self, state):
        return tf.concat([state['stoch'], state['deter']], -1)

    def get_dist(self, state):
        return tfd.MultivariateNormalDiag(state['mean'], state['std'])

    @tf.function
    def obs_step(self, prev_state, prev_action, embed):
        prior = self.img_step(prev_state, prev_action)  # get distribution+ of the current state
        x = tf.concat([prior['deter'], embed], -1)
        x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)
        x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = self.get_dist({'mean': mean, 'std': std}).sample()
        post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}  # get distr+ of next state
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



class RSSM_norm(tools.Module):

  def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
    super().__init__()
    self._name = "rssm"
    self._activation = act
    self._stoch_size = stoch
    self._deter_size = deter
    self._hidden_size = hidden
    self._cell = tfkl.GRUCell(self._deter_size)
    self._normalizer = tfkl.BatchNormalization(axis=-1)

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

  def get_feat(self, state, training=False):
    state = tf.concat([state['stoch'], state['deter']], -1)
    return state


  def get_dist(self, state):
    return tfd.MultivariateNormalDiag(state['mean'], state['std'])

  @tf.function
  def obs_step(self, prev_state, prev_action, embed):
    prior = self.img_step(prev_state, prev_action)  # get distribution+ of the current state
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}  # get distr+ of next state
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



class LidarEncoder(tools.Module):
    def __init__(self, output_dim, act=tf.nn.relu, obs_type="lidar"):
        self._name = "encoder"
        self._act = act
        self._output_dim = output_dim
        self._obs_type = obs_type

    def __call__(self, obs):
        kwargs = dict(strides=3, activation=self._act, padding='same')
        obs = obs[self._obs_type]
        if len(obs.shape) > 2:
            x = tf.reshape(obs, shape=(-1, *x.shape[2:], 1))
        else:
            x = tf.expand_dims(obs, axis=-1)
        x = self.get('conv1', tfkl.Conv1D, filters=4, kernel_size=5, **kwargs)(x)
        x = self.get('conv2', tfkl.Conv1D, filters=8, kernel_size=3, **kwargs)(x)
        x = self.get('flat', tfkl.Flatten)(x)

        shape = (*obs.shape[:-1], *x.shape[1:])
        return tf.reshape(x, shape=shape)


class MLPLidarEncoder(tools.Module):
    def __init__(self, encoded_dim, depth, act=tf.nn.relu):
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
    # note: this is just to see what happens without reducing the observation
    def __init__(self):
        self._name = "encoder"

    def __call__(self, obs):
        if type(obs) == dict:
            lidar = obs['lidar']
        else:
            lidar = obs
        return lidar


class MLPLidarDecoder(tools.Module):
    def __init__(self, depth, shape, act=tf.nn.relu):
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


class LidarDecoder(tools.Module):

    def __init__(self, output_dim, act=tf.nn.relu):
        self._name = "decoder"
        self._act = act
        self._output_dim = output_dim

    def __call__(self, features):
        params = tfpl.IndependentNormal.params_size(self._output_dim[0])
        x = tf.reshape(features, shape=(-1, *features.shape[2:]))
        x = self.get('params', tfkl.Dense, params, activation=self._act)(x)
        x = self.get('dist', tfpl.IndependentNormal, event_shape=self._output_dim[0])(x)
        dist = tfd.BatchReshape(x, batch_shape=features.shape[:2])
        return dist


class ConvEncoder(tools.Module):

    def __init__(self, depth=32, act=tf.nn.relu, obs_type="image"):
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
        if self._dist == 'binary':
            return tfd.Independent(tfd.Bernoulli(x), len(self._shape))
        raise NotImplementedError(self._dist)


class ActionDecoder(tools.Module):

  def __init__(
          self, size, layers, units, dist='tanh_normal', act=tf.nn.elu,
          min_std=1e-4, init_std=5, mean_scale=5):
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
    elif self._dist == 'normalized_tanhtransformed_normal':  # Like in dreamer, but with linear mean
        x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
        x = tf.reshape(x, [-1, 2 * self._size])
        x = self.get(f'hnorm', tfkl.BatchNormalization)(x, training=training)  # `training` true only in imagination
        x = tf.reshape(x, [*features.shape[:-1], -1])
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + self._min_std
        dist = tfd.Normal(mean, std)
        dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
        dist = tfd.Independent(dist, 1)
        dist = tools.SampleDist(dist)
    elif self._dist == 'normalized_tanh_normal':
        x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
        x = tf.reshape(x, [-1, 2 * self._size])
        x = self.get(f'hnorm', tfkl.BatchNormalization)(x, training=training)  # `training` true only in imagination
        x = tf.reshape(x, [*features.shape[:-1], -1])
        mean, std = tf.split(x, 2, -1)
        mean = tf.tanh(mean)  # action in +-1
        std = tf.nn.softplus(std) + self._min_std  # std is always positive
        dist = tfd.Normal(mean, std)
        dist = tfd.Independent(dist, 1)
    elif self._dist == 'normalized_linear_normal':
        x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
        x = tf.reshape(x, [-1, 2 * self._size])
        x = self.get(f'hnorm', tfkl.BatchNormalization)(x, training=training)  # `training` true only in imagination
        x = tf.reshape(x, [*features.shape[:-1], -1])
        mean, std = tf.split(x, 2, -1)
        mean = tf.clip_by_value(mean, -1, +1)
        std = tf.nn.softplus(std) + self._min_std
        dist = tfd.Normal(mean, std)
        dist = tfd.Independent(dist, 1)
    else:
        raise NotImplementedError(self._dist)
    return dist

