import collections
import time
import warnings
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from structs import PALETTE, SHORT_TRACKS_DICT, LONG_TRACKS_DICT, ALL_METHODS_DICT, BEST_MFREE_PERFORMANCES

DREAMER_CONFS = {}

Run = collections.namedtuple('Run', 'logdir train_track test_track method seed x y')
Trajectory = collections.namedtuple('Trajectory', 'logdir train_track test_track method seed x y v')

def parse_file(dir, file, file_parsers):
    filepath = file.relative_to(dir).parts[:-1][0]
    for file_parser in file_parsers:
        try:
            train_track, method, seed = file_parser(filepath)
            return train_track, method, seed
        except Exception as ex:     # if a parser fails, try to use the next one
            continue
    raise Warning(f'cannot parse {file}')   # if all the parsers fail, then raise expection



def check_track(track, tracks):
    return track=="" or track in tracks

def check_method(method, methods):
    return any([m in method for m in methods])

def get_tf_data(event_acc, method, tag):
    x, y = np.array([]), np.array([])
    if tag in event_acc.Tags()['tensors']:
        y = np.array([float(tf.make_ndarray(tensor.tensor_proto)) for tensor in event_acc.Tensors(tag)])
        x = np.array([tensor.step for tensor in event_acc.Tensors(tag)])
    elif tag + "_mean" in event_acc.Tags()['tensors']:
        y = np.array([float(tf.make_ndarray(tensor.tensor_proto)) for tensor in event_acc.Tensors(tag + "_mean")])
        x = np.array([tensor.step for tensor in event_acc.Tensors(tag + "_mean")])
    elif tag in event_acc.Tags()['scalars']:
        y = np.array([float(scalar.value) for scalar in event_acc.Scalars(tag)])
        x = np.array([int(scalar.step) for scalar in event_acc.Scalars(tag)])
    elif tag + "_mean" in event_acc.Tags()['scalars']:
        y = np.array([float(scalar.value) for scalar in event_acc.Scalars(tag + "_mean")])
        x = np.array([int(scalar.step) for scalar in event_acc.Scalars(tag + "_mean")])
    if 'sac' in method or 'ppo' in method:  # because we cannot scale steps in sb3 based on action_repeat
        x = x * 4
    return x, y


def load_runs(args, file_parsers, tag, eval_mode=False):
  runs = []
  init = time.time()
  for dir in args.indir:
    print(f'Loading runs from {dir}', end='\n')
    for file in dir.glob('**/events*'):
        print(f"\t[Time:{time.time()-init:.2f}] {file}")
        try:
            train_track, method, seed = parse_file(dir, file, file_parsers)
            if not check_track(train_track, args.tracks) or not check_method(method, args.methods):
                continue
            event_acc = EventAccumulator(str(file), size_guidance={'scalars': 1000,
                                                                   'tensors': 1000})  # max number of items to keep
            event_acc.Reload()
        except Warning as w:
            warnings.warn(w)
            continue
        except Exception as err:
            print(f'Error {file}: {err}')
            continue
        if eval_mode:
            for test_track in args.tracks:
                x, y = get_tf_data(event_acc, method, tag=f'{test_track}/{tag}')
                if args.first_n_models is not None:
                    eval_episodes = 10
                    x = x[:eval_episodes * args.first_n_models]    # to make uniform the eval, consider the same n of models
                    y = y[:eval_episodes * args.first_n_models]    # for each algorithm
                if x.shape[0] > 0 and y.shape[0] > 0:
                    runs.append(Run(file, train_track, test_track, method, seed, x, y))
        else:
            x, y = get_tf_data(event_acc, method, tag)
            if x.shape[0] > 0 and y.shape[0]>0:
                runs.append(Run(file, train_track, train_track, method, seed, x, y))  # in this case, train track = test track
    print()
  return runs


def load_trajectories(args, file_parsers):
  trajectories = []
  for dir in args.indir:
    print(f'Loading runs from {dir}', end='\n')
    for file in dir.glob('**/trajectory*checkpoint1*'):
        try:
            train_track, method, seed = parse_file(dir, file, file_parsers)
            test_track = file.parts[-1].split("_")[2]
            if not check_track(train_track, args.tracks) or not check_method(method, args.methods):
                continue
            trajectory = np.load(file)
            x, y, v = trajectory['position'][:, 0], trajectory['position'][:, 1], trajectory['velocity']
        except Warning as w:
            warnings.warn(w)
            continue
        except Exception as err:
            print(f'Error {file}: {err}')
            continue
        if x.shape[0] > 0 and y.shape[0] > 0 and v.shape[0] > 0:
            trajectories.append(Trajectory(file, train_track, test_track, method, seed, x, y, v))
        print('.', end='')
    print()
  return trajectories

def aggregate_max(runs):
    all_x = np.concatenate([r.x for r in runs])
    all_y = np.concatenate([r.y for r in runs])
    all_logs = np.concatenate([[r.logdir for _ in r.y] for r in runs])
    order = np.argsort(all_y)
    all_x, all_y, all_logs = all_x[order], all_y[order], all_logs[order]
    return all_x, all_y, all_logs


def plot_filled_curve(args, runs, axes, aggregator):
    tracks = sorted(set([r.train_track for r in runs]))
    methods = sorted(set([r.method for r in runs]))
    if not type(axes) == np.ndarray:
        axes = [axes]
    for i, (track, ax) in enumerate(zip(tracks, axes)):
        if args.show_labels:
            ax.set_title(LONG_TRACKS_DICT[track].upper())
            ax.set_xlabel(args.xlabel)
            if i <= 0:  # show y label only on first row
                ax.set_ylabel(args.ylabel if args.ylabel else args.tag)
        track_runs = [r for r in runs if r.train_track == track]
        for j, method in enumerate(methods):
            color = PALETTE[j]
            filter_runs = [r for r in track_runs if r.method == method]
            if len(filter_runs) > 0:
                x, mean, min, max = aggregator(filter_runs, args.binning)
                min = np.where(min > 0, min, 0)
                ax.plot(x, mean, color=color, label=method.upper())
                ax.fill_between(x, min, max, color=color, alpha=0.1)
        # plot baselines
        min_x = np.min(np.concatenate([r.x for r in track_runs]))
        max_x = np.max(np.concatenate([r.x for r in track_runs]))
        for j, (value, name) in enumerate(zip(args.hbaseline_values, args.hbaseline_names)):
            color = 'red'
            ax.hlines(y=value, xmin=min_x, xmax=max_x, color=color, linestyle='dotted', label=name.upper())
        if args.show_mfree_baselines:
            for j, (name, value) in enumerate(BEST_MFREE_PERFORMANCES[track].items()):
                color = PALETTE[len(methods) + len(args.hbaseline_values) + j]
                ax.hlines(y=value, xmin=min_x, xmax=max_x, color=color, linestyle='dashed', label=name.upper())
