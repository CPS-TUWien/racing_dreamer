import argparse
import pathlib
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from dreamer.plotting.log_parsers import DreamerParser
from dreamer.plotting.utils import load_runs

from dreamer.plotting.structs import ALL_METHODS_DICT, LONG_TRACKS_DICT, COLORS


def get_base_methods(methods):
  #methods format: e.g. `dreamer+distance_h_N`, `dreamer+occupancy_h_N`
  return list(set([method.split("_")[0] for method in methods]))

def plot_performance(args, tag, runs, axes):
  tracks = sorted(set([r.train_track for r in runs]))
  methods = sorted(set([r.method for r in runs]), key=lambda met: int(met.split("_")[-1]))
  basemethods = get_base_methods(methods)
  if not type(axes) == np.ndarray:
    axes = [axes]
  for i, (track, ax) in enumerate(zip(tracks, axes)):
    if args.show_labels:
      #ax.set_title(LONG_TRACKS_DICT[track].upper())
      ax.set_xlabel(args.xlabel)
      if i <= 0:  # show y label only on first row
        ax.set_ylabel(args.ylabel if args.ylabel else tag)
    track_runs = [r for r in runs if r.train_track == track]
    for j in range(len(basemethods)-1, -1, -1):    # to have the second line plotted behind
      basemethod = basemethods[j]
      xmethods, best_performances_means, best_performances_stds = [], [], []
      first_filter = [r for r in track_runs if basemethod in r.method]
      for k, method in enumerate(methods):
        filter_runs = [r for r in first_filter if r.method == method]
        if len(filter_runs) > 0:
          maxes = [np.max(r.y) for r in filter_runs]
          mean, std = np.nanmean(maxes), np.nanstd(maxes)
          best_performances_means.append(mean)
          best_performances_stds.append(std)
          xmethods.append(int(method.split("_")[-1]))
      fmt = COLORS[basemethod]
      capsz = 2 if j%2==0 else 4
      best_performances_means = np.stack(best_performances_means)
      best_performances_stds = np.stack(best_performances_stds)
      ax.errorbar(xmethods, best_performances_means, yerr=best_performances_stds, barsabove=True,
                  fmt=fmt, capsize=capsz, label=basemethod.upper())
    ax.set_xticks(xmethods)
    ax.set_xticklabels(xmethods)
    ax.set_ylim(0.25, 1.5)
    # keep only axis, remove top/right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

from collections.abc import Iterable   # import directly from collections for Python < 3.3

def main(args):
  assert len(args.hbaseline_names) == len(args.hbaseline_values)
  tag = "test/progress"   # load_run will check for `tag` or `tag_mean`
  gby_param = 'horizon'
  runs = load_runs(args, [DreamerParser(gby_parameter=gby_param)], tag=tag)
  tracks = sorted(set([r.train_track for r in runs]))
  args.outdir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  fig, axes = plt.subplots(1, len(tracks), figsize=(6 * len(tracks), 3))
  plot_performance(args, tag, runs, axes)
  if args.legend:
    if not type(axes) == np.ndarray:  # in case of fig with a single axis
      axes = [axes]
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               framealpha=1.0, fontsize='small')
  filename = f'curves_' + '_'.join(tracks) + f'_compare_{gby_param}_{timestamp}.png'
  fig.tight_layout(pad=1.0)
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
  parser.add_argument('--methods', nargs='+', type=str, default=ALL_METHODS_DICT.keys())
  parser.add_argument('--hbaseline_names', nargs='+', type=str, default=[])
  parser.add_argument('--hbaseline_values', nargs='+', type=float, default=[])
  return parser.parse_args()


if __name__=='__main__':
  init = time.time()
  main(parse())
  print(f"\n[Info] Elapsed Time: {time.time()-init:.3f} seconds")