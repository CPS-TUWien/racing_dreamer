import argparse
import time
import pathlib
import warnings
from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dreamer.plotting.aggregators import MeanStd, MeanMinMax
from dreamer.plotting.log_parsers import EvaluationParser
from dreamer.plotting.plot_test_evaluation import plot_error_bar
from dreamer.plotting.structs import LONG_TRACKS_DICT, ALL_METHODS_DICT, FONTSIZE
import matplotlib.pyplot as plt
import numpy as np

from dreamer.plotting.utils import parse_file, check_track, check_method, get_tf_data, Run


def load_filtered_runs(args, file_parsers, tag, filter_tag, threshold):
  runs = []
  for dir in args.indir:
    print(f'Loading runs from {dir}', end='')
    for file in dir.glob('**/events*'):
        try:
            train_track, method, seed = parse_file(dir, file, file_parsers)
            if not check_track(train_track, args.tracks) or not check_method(method, args.methods):
                continue
            event_acc = EventAccumulator(str(file), size_guidance={'scalars': 100000,
                                                                   'tensors': 100000})  # max number of items to keep
            event_acc.Reload()
        except Warning as w:
            warnings.warn(w)
            continue
        except Exception as err:
            print(f'Error {file}: {err}')
            continue
        for test_track in args.tracks:
            _, y = get_tf_data(event_acc, method, tag=f'{test_track}/{tag}')
            x, f = get_tf_data(event_acc, method, tag=f'{test_track}/{filter_tag}')
            if args.first_n_models is not None:
                eval_episodes = 10
                x = x[:eval_episodes * args.first_n_models]    # to make uniform the eval, consider the same n of models
                y = y[:eval_episodes * args.first_n_models]    # for each algorithm
                f = f[:eval_episodes * args.first_n_models]    # for each algorithm
            x = x[np.nonzero(f > threshold)]
            y = y[np.nonzero(f > threshold)]
            if x.shape[0] > 0 and y.shape[0] > 0:
                runs.append(Run(file, train_track, test_track.replace("_", ""), method, seed, x, y))
            print('.', end='')
    print()
  return runs

def main(args):
    tag = "time"
    args.ylabel = args.ylabel if args.ylabel != "" else tag
    runs = load_filtered_runs(args, [EvaluationParser()], tag='time', filter_tag='progress', threshold=0.95)
    train_tracks = sorted(set([r.train_track for r in runs if r.train_track!=""]))
    args.outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for aggregator, fn in zip(['mean_minmax'], [MeanMinMax()]):
        fig, axes = plt.subplots(1, len(train_tracks), figsize=(4 * len(train_tracks), 3))
        # todo move loop on train tracks in plot error bar
        for i, (train_track, ax) in enumerate(zip(train_tracks, axes)):
            filter_runs = [r for r in runs if r.train_track == train_track or r.train_track==""]
            plot_error_bar(args, filter_runs, ax, aggregator=fn)
        if args.legend:
            if not type(axes) == np.ndarray:  # in case of fig with a single axis
                axes = [axes]
            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(labels), framealpha=1.0,
                       handletextpad=0.1, fontsize=FONTSIZE, columnspacing=0.2)
        filename = f'time_' + '_'.join(train_tracks) + f'_{aggregator}_{timestamp}.pdf'
        fig.tight_layout(pad=2.5)
        plt.subplots_adjust(bottom=0.25)
        fig.savefig(args.outdir / filename)
        print(f"[Info] Written {args.outdir / filename}")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', nargs='+', type=pathlib.Path, required=True)
    parser.add_argument('--outdir', type=pathlib.Path, required=True)
    parser.add_argument('--xlabel', type=str, default="")
    parser.add_argument('--ylabel', type=str, default="")
    parser.add_argument('--legend', action='store_true')
    parser.add_argument('--show_labels', action='store_true')
    parser.add_argument('--tracks', nargs='+', type=str, default=LONG_TRACKS_DICT.keys())
    parser.add_argument('--vis_tracks', nargs='+', type=str, default=LONG_TRACKS_DICT.keys())
    parser.add_argument('--methods', nargs='+', type=str, default=ALL_METHODS_DICT.keys())
    parser.add_argument('--first_n_models', type=int, required=None, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    init = time.time()
    main(parse())
    print(f"\n[Info] Elapsed Time: {time.time() - init:.3f} seconds")
