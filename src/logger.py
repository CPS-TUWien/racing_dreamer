import os
from glob import glob
from typing import Union, Any, Dict

from acme.utils.counting import Counter
from acme.utils.loggers import Logger, LoggingData
import tensorflow as tf
import numpy as np

class TensorboardLogger(Logger):
    def __init__(self, logdir: str, prefix: str = None):
        self._prefix = prefix
        self._counter = Counter()
        self._writer = tf.summary.create_file_writer(logdir)

    def write(self, data: LoggingData, step: int = 1, prefix: str = None):
        prefix = self._prefix if not prefix else prefix
        with self._writer.as_default():
            for key, value in data.items():
                label = f'{prefix}/{key}' if prefix else key
                self._counter.increment(**{label: step})
                current_step = self._counter.get_counts()[label]
                self._write(label=label, data=value, step=current_step)

    def _write(self, label: str, data: Any, step: int):
        if isinstance(data, tf.Tensor):
            self._write_tensor(label, data, step)
        elif isinstance(data, dict):
            self._write_dict(label, data, step)

    def _write_tensor(self, label: str, data: tf.Tensor, step: int):
        if np.isscalar(data) or len(data.shape) == 0:
            tf.summary.scalar(name=label, data=data, step=step)
        elif len(data.shape) == 4:
            tf.summary.image(name=label, data=data, step=step)

    def _write_dict(self, label: str, data: Dict, step: int):
        self.write(data=data, prefix=label, step=step)
