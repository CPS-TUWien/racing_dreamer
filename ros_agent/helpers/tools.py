import datetime
import io
import math
import pathlib
import pickle
import re
import uuid
import imageio
import gym
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp
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

  def _forward_log_det_jacobian(self, x):
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
      print(f'Found {count} {self._name} parameters.')
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
        strategy.extended.update(var, lambda var: self._wd * var)


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



