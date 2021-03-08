import argparse
import time
import pathlib
from datetime import datetime

from dreamer.plotting.aggregators import MeanMinMax
from dreamer.plotting.log_parsers import EvaluationParser
from dreamer.plotting.plot_training_curves import sort_methods
from dreamer.plotting.structs import LONG_TRACKS_DICT, ALL_METHODS_DICT, SHORT_TRACKS_DICT, COLORS, FONTSIZE
from dreamer.plotting.utils import load_runs
import matplotlib.pyplot as plt
import numpy as np


def plot_error_bar(args, runs, ax, aggregator):
    train_track = sorted(set([r.train_track for r in runs if r.train_track!=""]))
    assert len(train_track)==1
    train_track = train_track[0]
    methods = sorted(set([r.method for r in runs]))
    bar_width = 1 / (len(methods)+1)
    interbar = (1 - (1 / (len(methods)+1))) / 30
    for i, method in enumerate(sorted(methods, key=sort_methods)):
        means, n_errors, p_errors, colors, ecolors, test_tracks = [], [], [], [], [], []
        for j, test_track in enumerate(args.vis_tracks):
            filter_runs = [r for r in runs if r.test_track == test_track and r.method == method]
            if len(filter_runs) > 0:
                x, mean, min, max = aggregator(filter_runs, None)
                means.append(mean[0])
                n_errors.append(mean[0] - min[0])
                p_errors.append(max[0] - mean[0])
            else:
                means.append(0.0)
                n_errors.append(0.0)
                p_errors.append(0.0)
            if method in COLORS.keys():
                colors.append(COLORS[method])
            else:
                colors.append('black')
            test_tracks.append(test_track)
        xpos = np.arange(1, len(test_tracks) + 1)
        one_label = False
        for x, m, ne, pe, c in zip(xpos, means, n_errors, p_errors, colors):
            if j == len(args.vis_tracks)-1 and not one_label:
                ax.bar(x + i * (bar_width + interbar), m, bar_width, yerr=np.array([ne, pe]).reshape((2,1)),
                       align='center', alpha=0.7, color=c, ecolor=c, capsize=3, label=method.upper())
                one_label = True
            else:
                ax.bar(x + i * (bar_width + interbar), m, bar_width, yerr=np.array([ne, pe]).reshape((2, 1)),
                       align='center', alpha=0.7, color=c, ecolor=c, capsize=3)
    mins = xpos - bar_width / 2
    ax.set_xticks((mins + (mins + len(methods) * (bar_width+interbar))) / 2)
    ax.set_xticklabels([SHORT_TRACKS_DICT[track] for track in test_tracks])
    ax.set_title(f'TRAINED ON {LONG_TRACKS_DICT[train_track]}'.upper())
    ax.set_ylabel(args.ylabel)
    #ax.set_ylim(0, 1.1)
    # keep only axis, remove top/right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(FONTSIZE)


def main(args):
    tag = "progress"
    args.ylabel = args.ylabel if args.ylabel != "" else tag
    runs = load_runs(args, [EvaluationParser()], tag=tag, eval_mode=True)
    train_tracks = sorted(set([r.train_track for r in runs if r.train_track != ""]))
    args.outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for aggregator, fn in zip(['mean_minmax'], [MeanMinMax()]):
        fig, axes = plt.subplots(1, len(train_tracks), figsize=(4 * len(train_tracks), 3))
        for i, (train_track, ax) in enumerate(zip(train_tracks, axes)):
            filter_runs = [r for r in runs if r.train_track == "" or r.train_track==train_track]
            plot_error_bar(args, filter_runs, ax, aggregator=fn)
        if args.legend:
            if not type(axes) == np.ndarray:  # in case of fig with a single axis
                axes = [axes]
            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(labels), framealpha=1.0,
                       handletextpad=0.1, fontsize=FONTSIZE, columnspacing=0.2)
        filename = f'eval_' + '_'.join(train_tracks) + f'_{aggregator}_{timestamp}.pdf'
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
    parser.add_argument('--tracks', nargs='+', type=str, default=LONG_TRACKS_DICT.keys())
    parser.add_argument('--vis_tracks', nargs='+', type=str, default=LONG_TRACKS_DICT.keys())
    parser.add_argument('--methods', nargs='+', type=str, default=ALL_METHODS_DICT.keys())
    parser.add_argument('--first_n_models', type=int, required=None, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    init = time.time()
    main(parse())
    print(f"\n[Info] Elapsed Time: {time.time() - init:.3f} seconds")
