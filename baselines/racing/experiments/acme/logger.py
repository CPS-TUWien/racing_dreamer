import os
from glob import glob
from typing import Union, Any, Dict, List, Tuple

from acme.utils.counting import Counter
from acme.utils.loggers import Logger, LoggingData
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np

class TensorBoardLogger(Logger):
    def __init__(self, logdir: str, file_suffix: str = None):
        self._counter = Counter()
        self._writer = tf.summary.create_file_writer(logdir, filename_suffix=file_suffix)

    def write(self, data: LoggingData, incremental_step: int = 1, step: int = None):
        with self._writer.as_default():
            for key, value in data.items():
                if step is None:
                    step = self._counter.increment(**{key: incremental_step})[key]
                self._write(label=key, data=value, step=step)

    def _write(self, label: str, data: Any, step: int):
        if isinstance(data, tf.Tensor) or isinstance(data, np.ndarray) or isinstance(data, float) or isinstance(data, int):
            self._write_tensor(label, data, step)
        else:
            raise NotImplementedError(f'{type(data)} not supported')


    def _write_tensor(self, label: str, data: Union[np.ndarray, tf.Tensor], step: int):
        if np.isscalar(data) or len(data.shape) == 0:
            tf.summary.scalar(name=label, data=data, step=step)
        elif len(data.shape) == 4:
            tf.summary.image(name=label, data=data, step=step)

    def close(self):
        self._writer.close()


class PrefixedTensorBoardLogger(Logger):
    def __init__(self, base_logger: TensorBoardLogger, prefix: str):
        self._prefix = prefix
        self._logger = base_logger

    def write(self, data: LoggingData, step: int = None):
        self._logger.write(dict([
            (f'{self._prefix}/{key}', value)
            for key, value
            in data.items()
        ]), step=step)

    def close(self) -> None:
        self._logger.close()



class MetricLogger(Logger):
    def __init__(self):
        self._data = dict()

    def __getitem__(self, item):
        return self._data[item]

    def write(self, data: LoggingData, **kwargs):
        self._data.update(data)

class LoggerList(Logger):
    def __init__(self, loggers: List[Logger]):
        self._loggers = loggers

    def write(self, data: LoggingData, **kwargs):
        for logger in self._loggers:
            logger.write(data, **kwargs)

class HyperParameterLogger(Logger):

    def __init__(self, logdir: str, hparams: Dict[str, Union[Tuple[float, float], List]], metrics: Dict[str, str]):
        self._hparams = []
        for name, param in hparams.items():
            if isinstance(param, Tuple):
                min, max = param
                if isinstance(min, float):
                    self._hparams.append(hp.HParam(name, hp.RealInterval(min_value=min, max_value=max)))
                elif isinstance(min, int):
                    self._hparams.append(hp.HParam(name, hp.IntInterval(min_value=min, max_value=max)))
            elif isinstance(param, List):
                self._hparams.append(hp.HParam(name, hp.Discrete(param)))

        self._metrics = metrics
        self._writer = tf.summary.create_file_writer(logdir=logdir)
        with self._writer.as_default():
            hp.hparams_config(
                hparams=self._hparams,
                metrics=[hp.Metric(name, display_name=display) for name, display in metrics.items()],
            )

    def write(self, data: LoggingData, step: int = None):
        with self._writer.as_default():
            params = dict([(param, data[param.name]) for param in self._hparams])
            hp.hparams(params)
            for metric, display in self._metrics.items():
                tf.summary.scalar(name=metric, data=data[metric], step=step)