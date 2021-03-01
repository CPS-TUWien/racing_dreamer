import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import kerasncp as kncp
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from yamldataclassconfig.config import YamlDataClassConfig

import tools


@dataclass
class Config(YamlDataClassConfig):
    filters: List[int] = field(default_factory=lambda: [18, 20, 22])
    kernels: List[int] = field(default_factory=lambda: [10, 10, 10])
    strides: List[int] = field(default_factory=lambda: [3, 2, 2])
    encoded_dim: int = 32
    inter_neurons: int = 24  # Number of inter neurons
    command_neurons: int = 12  # Number of command neurons
    motor_neurons: int = 2  # Number of motor neurons
    sensory_fanout: int = 6  # How many outgoing synapses has each sensory neuron
    inter_fanout: int = 4  # How many outgoing synapses has each inter neuron
    recurrent_command_synapses: int = 6  # Now many recurrent synapses are in the command neuron layer
    motor_fanin: int = 4  # How many incoming synapses has each motor neuron


class RacingNCP(tools.Module):

    def __init__(self, checkpoint_dir: pathlib.Path):
        super().__init__()
        config = Config()
        if (checkpoint_dir / 'conv_ncp.yml').exists():
            config.load(checkpoint_dir / 'conv_ncp.yml')
        self._head = ConvHead(filters=config.filters, kernels=config.kernels,
                              strides=config.strides, encoded_dim=config.encoded_dim)
        self._ncp = NCP(inter_neurons=config.inter_neurons, command_neurons=config.command_neurons,
                        motor_neurons=config.motor_neurons, sensory_fanout=config.sensory_fanout,
                        inter_fanout=config.inter_fanout, recurrent_command_synapses=config.recurrent_command_synapses,
                        motor_fanin=config.motor_fanin)
        self.build_model()
        # load checkpoint
        if (checkpoint_dir / 'variables.pkl').exists():
            self.load(checkpoint_dir / 'variables.pkl')
            print('Load checkpoint.')
        else:
            raise FileNotFoundError(f"checkpoint {checkpoint_dir / 'variables.pkl'} missing")

    def __call__(self, inputs, **kwargs):
        embed_reshape = tf.concat([tf.shape(inputs)[:-2], [32]], 0)
        x = tf.reshape(inputs, (-1,) + tuple(inputs.shape[-2:]))  # apply same conv to all B*T observations
        embeds = self._head(x)
        embeds = tf.reshape(embeds, embed_reshape)
        return self._ncp(embeds)  # rnn layer takes input of shape (B, T, *)

    def build_model(self):
        dummy_input = tf.zeros([1, 1, 1080, 1])
        self(dummy_input)

    @staticmethod
    def _preprocess_lidar(scan):
        # Step 1: clip values in simulated sensors' ranges
        min_range, max_range = 0.0, 15.0
        lidar = np.clip(scan, min_range, max_range)
        # Step 2: normalize lidar ranges in 0, 1
        lidar = (lidar - min_range) / (max_range - min_range) - 0.5
        lidar = np.reshape(lidar, [1, 1080, 1]).astype('float32')
        return lidar

    @staticmethod
    def _postprocess_action(action):
        clip = lambda x, l, u: max(l, min(u, x))
        motor = clip(action[0].numpy(), 0.005, 1.0)  # avoid negative value from ncp agent
        steering = clip(action[1].numpy(), -1.0, 1.0)  # avoid angles > 1
        dict_action = {'motor': motor, 'steering': steering}  # compatibility with racecar_gym action format
        return dict_action

    def action(self, observation: Dict[str, np.ndarray], prev_state: Tuple[tf.Tensor, tf.Tensor] = None):
        scan = observation['lidar']
        scan = self._preprocess_lidar(scan)
        embed = self._head(scan)

        if prev_state is None:
            state = tf.zeros((1, self._ncp._ltc.state_size))
        else:
            _, state = prev_state
        action, state = self._ncp._ltc(embed, [state])
        action = self._postprocess_action(action[0])  # note: in state action must have 2 dims for shape matching
        return action, (embed, state[0])


class NCP(tools.Module):

    def __init__(self, inter_neurons, command_neurons, motor_neurons, sensory_fanout, inter_fanout,
                 recurrent_command_synapses, motor_fanin):
        super().__init__()
        self._name = 'ncp_layer'
        ncp_arch = kncp.wirings.NCP(
            inter_neurons=inter_neurons,
            command_neurons=command_neurons,
            motor_neurons=motor_neurons,
            sensory_fanout=sensory_fanout,
            inter_fanout=inter_fanout,
            recurrent_command_synapses=recurrent_command_synapses,
            motor_fanin=motor_fanin
        )
        self._wirings = ncp_arch
        self._ltc = kncp.LTCCell(ncp_arch)
        self._ncp_cell = tfkl.RNN(self._ltc, return_sequences=False)  # for inference, return single pred

    def __call__(self, inputs, **kwargs):
        return self._ncp_cell(inputs, **kwargs)


class ConvHead(tools.Module):
    def __init__(self, filters, kernels, strides, encoded_dim):
        super().__init__()
        self._name = 'conv_head'
        self._encoded_dim = encoded_dim
        self._act = 'relu'
        self._filters = filters
        self._kernels = kernels
        self._strides = strides

    def __call__(self, x):
        for i, (filters, kernel, stride) in enumerate(zip(self._filters, self._kernels, self._strides)):
            x = self.get(f'conv{i + 1}', tfkl.Conv1D, filters=filters, kernel_size=kernel, strides=stride,
                         activation=self._act)(x)
            if i > 0 and i % 2 == 1:  # every 2 conv layers, put a max pool layer
                x = self.get(f'max-pool{i}', tfkl.MaxPool1D)(x)
        x = self.get('flat', tfkl.Flatten)(x)
        x = self.get('dense', tfkl.Dense, units=self._encoded_dim, activation=self._act)(x)
        return x
