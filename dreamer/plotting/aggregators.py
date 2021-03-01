import warnings
from typing import Tuple

import numpy as np
from abc import abstractmethod


class Aggregator:

    @property
    @abstractmethod
    def reducer(self):
        pass

    def __call__(self, runs, binning, minx=0, maxx=8000000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate data according to the given binning. If `binning` is None aggregate over the all range.
        :param runs:    list of Run structs
        :param binning: binning size
        :return:        tuple (x, mean, min, max)
        """
        all_x, all_y = np.concatenate([r.x for r in runs]), np.concatenate([r.y for r in runs])
        order = np.argsort(all_x)
        all_x, all_y = all_x[order], all_y[order]
        if binning is None:
            binned_x = [np.max(all_x)]
        else:
            minx = max(minx, all_x.min())
            maxx = min(maxx, all_x.max()+binning)
            binned_x = np.arange(minx, maxx, binning)
        binned_mean, binned_min, binned_max = [], [], []
        for start, stop in zip([-np.inf] + list(binned_x), list(binned_x)):
            left = (all_x <= start).sum()
            right = (all_x <= stop).sum()
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    mean, min_margin, max_margin = self.reducer(all_y[left:right])
                except RuntimeWarning as wrn:
                    print(f'[WARNING] {wrn}. Consider to increase the binning')
                    exit(1)
            binned_mean.append(mean)
            binned_min.append(min_margin)
            binned_max.append(max_margin)
        return np.array(binned_x), np.array(binned_mean), np.array(binned_min), np.array(binned_max)


class MeanStd(Aggregator):
    @property
    def reducer(self):
        return lambda x: (np.nanmean(np.array(x)), np.nanmean(np.array(x)) - np.nanstd(np.array(x)),
                          np.nanmean(np.array(x)) + np.nanstd(np.array(x)))


class MeanMinMax(Aggregator):
    @property
    def reducer(self):
        return lambda x: (np.nanmean(np.array(x)), np.nanmin(np.array(x)), np.nanmax(np.array(x)))
