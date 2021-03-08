import argparse
import time
import pathlib

from matplotlib.collections import LineCollection
from racecar_gym.bullet.configs import SceneConfig
from racecar_gym.bullet.providers import resolve_path
from racecar_gym.core.gridmaps import GridMap

from dreamer.plotting.log_parsers import EvaluationParser
from dreamer.plotting.structs import LONG_TRACKS_DICT, ALL_METHODS_DICT, SHORT_TRACKS_DICT, COLORS
from dreamer.plotting.utils import load_runs, load_trajectories
import matplotlib.pyplot as plt
import numpy as np


def load_map(track):
    if "treitlstrasse" in track:
        track = "treitlstrasse_v2"
    config_file = f'/home/luigi/PycharmProjects/racecar_gym/models/scenes/{track}/{track}.yml'
    config = SceneConfig()
    config.load(config_file)
    config.sdf = resolve_path(file=config_file, relative_path=config.sdf)
    config.map.maps = resolve_path(file=config_file, relative_path=config.map.maps)
    config.map.starting_grid = resolve_path(file=config_file, relative_path=config.map.starting_grid)
    maps = dict([
        (name, GridMap(grid_map=np.load(config.map.maps)[data],
                       origin=config.map.origin,
                       resolution=config.map.resolution
                       ))
        for name, data
        in [
            ('progress', 'norm_distance_from_start'),
            ('obstacle', 'norm_distance_to_obstacle'),
            ('occupancy', 'drivable_area')
        ]
    ])
    return maps['occupancy']


def plot_trajectories(trajectories, agent, train_track, test_track, outdir, minvel=0.5, maxvel=4.0, margin=50):
    map = load_map(test_track)
    nonzero = np.argwhere(map.map > 0)
    miny, minx = np.min(nonzero, axis=0)
    maxy, maxx = np.max(nonzero, axis=0)

    fig, ax = plt.subplots(1, 1)
    norm = plt.Normalize(minvel, maxvel)    # for colorbar
    xx, yy, vv = [t.x for t in trajectories], [t.y for t in trajectories], [t.v for t in trajectories]

    rr = [[map.to_pixel((x, y))[0] for x, y in zip(t.x, t.y)] for t in trajectories]
    cc = [[map.to_pixel((x, y))[1] for x, y in zip(t.x, t.y)] for t in trajectories]

    #map = np.flip(map.map, axis=0)
    map = map.map
    #minx, maxx = np.min(np.concatenate(cc)), np.max(np.concatenate(cc))
    #miny, maxy = np.min(np.concatenate(rr)), np.max(np.concatenate(rr))
    # transform the map
    #map = map[miny - margin:maxy + margin, minx - margin:maxx + margin]
    map = map[miny - margin:maxy + margin, minx - margin:maxx + margin]
    map = np.where(map, 150, 255)
    map = np.stack([map, map, map], axis=-1)
    ax.imshow(map)
    for x, y, v in zip(cc, rr, vv):
        points = np.array([x - (minx - margin), y - (miny - margin)]).T.reshape(-1, 1, 2)
        #points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        # Set the values used for colormapping
        lc.set_array(v)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
    # keep only axis, remove top/right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # add colorbar
    fig.colorbar(line, ax=ax)
    plt.axis('equal')
    suffix = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    filename = f"{outdir}/trajectory_{agent}_{train_track}_{test_track}_{suffix}.pdf"
    fig.savefig(filename)
    print(f"[Info] Written {filename}")


def plot_trajectories_learned_methods(trajectories, outdir):
    methods = list(set([t.method for t in trajectories]))
    train_tracks = set([t.train_track for t in trajectories])
    test_tracks = set([t.test_track for t in trajectories])
    for k, method in enumerate(methods):
        for i, train_track in enumerate(train_tracks):
            for j, test_track in enumerate(test_tracks):
                filter_trajectories = [t for t in trajectories if t.method == method and t.train_track == train_track
                                       and t.test_track == test_track]
                if len(filter_trajectories) > 0:
                    plot_trajectories(filter_trajectories, method, train_track, test_track, outdir)


def plot_trajectories_programmed_methods(trajectories, outdir):
    methods = list(set([t.method for t in trajectories]))
    train_tracks = list(set([t.train_track for t in trajectories]))
    assert len(train_tracks) == 1 and train_tracks[0] == ""
    test_tracks = list(set([t.test_track for t in trajectories]))
    for k, method in enumerate(methods):
        for i, test_track in enumerate(test_tracks):
            filter_trajectories = [t for t in trajectories if t.method == method and t.test_track == test_track]
            if len(filter_trajectories) > 0:
                plot_trajectories(filter_trajectories, method, train_tracks[0], test_track, outdir)


def main(args):
    trajs = load_trajectories(args, [EvaluationParser()])
    learned_methods_runs = [t for t in trajs if "ftg" not in t.method]
    programmed_methods_runs = [t for t in trajs if "ftg" in t.method]
    suffix = str(time.time())
    if len(learned_methods_runs) > 0:
        plot_trajectories_learned_methods(learned_methods_runs, args.outdir / suffix)
    if len(programmed_methods_runs) > 0:
        plot_trajectories_programmed_methods(programmed_methods_runs, args.outdir / suffix)


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
    return parser.parse_args()


if __name__ == '__main__':
    init = time.time()
    main(parse())
    print(f"\n[Info] Elapsed Time: {time.time() - init:.3f} seconds")
