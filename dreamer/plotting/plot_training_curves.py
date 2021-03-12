import argparse
from datetime import datetime
import pathlib
import time
import matplotlib.pyplot as plt
import numpy as np

from aggregators import MeanStd
from utils import load_runs
from log_parsers import ModelFreeParser, DreamerParser

from structs import LONG_TRACKS_DICT, ALL_METHODS_DICT, BEST_MFREE_PERFORMANCES, \
    BEST_DREAMER_PERFORMANCES, COLORS, FONTSIZE


def sort_methods(method):
    for i, m in enumerate(ALL_METHODS_DICT.keys()):
        if m in method:
            return i
    return np.Inf

def plot_filled_curve(args, runs, axes, aggregator):
    tracks = sorted(set([r.train_track for r in runs]))
    methods = sorted(set([r.method for r in runs]))
    if not type(axes) == np.ndarray:
        axes = [axes]
    for i, (track, ax) in enumerate(zip(tracks, axes)):
        ax.set_title(LONG_TRACKS_DICT[track].upper())
        ax.set_xlabel(args.xlabel)
        if i <= 0:  # show y label only on first row
            ax.set_ylabel(args.ylabel)
        track_runs = [r for r in runs if r.train_track == track]
        for j, method in enumerate(sorted(methods, key=sort_methods)):
            if method in COLORS.keys():
                color = COLORS[method]
            else:
                color = '#fffff'
            filter_runs = [r for r in track_runs if r.method == method]
            if len(filter_runs) > 0:
                x, mean, min, max = aggregator(filter_runs, args.binning)
                min = np.where(min > 0, min, 0)
                ax.plot(x, mean, color=color, label=method.upper())
                ax.fill_between(x, min, max, color=color, alpha=0.1)
                print(
                    f"\t[Info] Track: {track}, Method: {method}, Mean Max Progress: {np.max(mean)}, Max progress: {np.max(max)}")
        # plot baselines
        min_x = np.min(np.concatenate([r.x for r in track_runs]))
        max_x = np.min([8000000, np.max(np.concatenate([r.x for r in track_runs]))])
        if args.show_mfree_baselines:
            for j, (name, value) in enumerate(BEST_MFREE_PERFORMANCES[track].items()):
                if name in COLORS.keys():
                    color = COLORS[name]
                else:
                    color = '#fffff'
                ax.hlines(y=value, xmin=min_x, xmax=max_x, color=color, linestyle='dashed', label=name.upper())
        elif args.show_dreamer_baselines:
            for j, (name, value) in enumerate(BEST_DREAMER_PERFORMANCES[track].items()):
                if name in COLORS.keys():
                    color = COLORS[name]
                else:
                    color = '#fffff'
                ax.hlines(y=value, xmin=min_x, xmax=max_x, color=color, linestyle='dashed', label=name.upper())
                ax.set_ylim(0, value+0.1)
        for j, (value, name) in enumerate(zip(args.hbaseline_values, args.hbaseline_names)):
            color = 'red'
            ax.hlines(y=value, xmin=min_x, xmax=max_x, color=color, linestyle='dotted', label=name.upper())
        # keep only axis, remove top/right
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(range(0, 8000001, 2000000))
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(FONTSIZE)


def main(args):
    assert len(args.hbaseline_names) == len(args.hbaseline_values)
    tag = "test/progress_mean"
    args.ylabel = args.ylabel if args.ylabel != "" else tag
    runs = load_runs(args, [DreamerParser(), ModelFreeParser()], tag=tag)
    tracks = sorted(set([r.train_track for r in runs]))
    args.outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for aggregator, fn in zip(['mean_std'], [MeanStd()]):
        fig, axes = plt.subplots(1, len(tracks), figsize=(max(8, 3 * len(tracks)), 5))
        plt.rc('font', size=FONTSIZE)          # controls default text sizes
        plot_filled_curve(args, runs, axes, aggregator=fn)
        if args.legend:
            if not type(axes) == np.ndarray:  # in case of fig with a single axis
                axes = [axes]
            handles, labels = axes[-1].get_legend_handles_labels()
            fontsize = FONTSIZE     #'x-small' if len(labels)>6 else 'small'
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.08, 0.00, 0.85, .27),
                   bbox_transform=fig.transFigure, mode='expand',
                   columnspacing=1, handletextpad=0.0, ncol=4, framealpha=1.0, fontsize=fontsize)
        filename = f'curves_' + '_'.join(tracks) + f'_{aggregator}_{timestamp}.pdf'
        fig.tight_layout(pad=1.0)
        fig.subplots_adjust(bottom=0.35)
        fig.savefig(args.outdir / filename)
        print(f"[Info] Written {args.outdir / filename}")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', nargs='+', type=pathlib.Path, required=True)
    parser.add_argument('--outdir', type=pathlib.Path, required=True)
    parser.add_argument('--xlabel', type=str, default="")
    parser.add_argument('--ylabel', type=str, default="")
    parser.add_argument('--binning', type=int, default=15000)
    parser.add_argument('--legend', action='store_true')
    parser.add_argument('--show_mfree_baselines', action='store_true')
    parser.add_argument('--show_dreamer_baselines', action='store_true')
    parser.add_argument('--tracks', nargs='+', type=str, default=LONG_TRACKS_DICT.keys())
    parser.add_argument('--methods', nargs='+', type=str, default=ALL_METHODS_DICT.keys())
    parser.add_argument('--hbaseline_names', nargs='+', type=str, default=[])
    parser.add_argument('--hbaseline_values', nargs='+', type=float, default=[])
    return parser.parse_args()


if __name__ == '__main__':
    init = time.time()
    main(parse())
    print(f"\n[Info] Elapsed Time: {time.time() - init:.3f} seconds")
