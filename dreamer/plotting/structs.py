DREAMER_PALETTE = 10 * ('#377eb8', '#4daf4a', '#cab2d6', '#fb9a99', '#fdbf6f')
PALETTE = 10 * ('#984ea3', '#e41a1c', '#ff7f00', '#a65628', '#f781bf', '#888888', '#a6cee3', '#b2df8a')

COLORS = {
    'dreamer+distance': '#377eb8', 'dreamer+occupancy': '#4daf4a', 'dreamer': '#377eb8',
    'd4pg': '#984ea3', 'mpo': '#e41a1c',
    'lstm-ppo': '#ff7f00', 'ppo': '#a65628',
    'sac': '#888888', 'ftg': '#fdbf6f'
}
FONTSIZE=14
OBSTYPE_DICT = {"lidar": "distance", "lidar_occupancy": "occupancy", "lidaroccupancy": "occupancy", }

LONG_TRACKS_DICT = {'austria': 'AUSTRIA', 'columbia': 'COLUMBIA',
                    'treitlstrasse': 'TREITLSTRASSE', 'treitlstrasse_v2': 'TREITLSTRASSE', 'treitlstrassev2': 'TREITLSTRASSE',
                    'barcelona': 'BARCELONA'}
SHORT_TRACKS_DICT = {'austria': 'AUT', 'columbia': 'COL',
                     'treitlstrasse': 'TRT', 'treitlstrasse_v2': 'TRT', 'treitlstrassev2': 'TRT',
                     'barcelona': 'BRC'}
ALL_METHODS_DICT = {'dream': 'Dream', 'd4pg': 'D4PG', 'mpo': 'MPO', 'lstm-ppo': 'LSTM-PPO',
                    'ppo': 'PPO', 'sac': 'SAC', 'ftg': 'FTG'}
ALL_VARIANTS_DICT = {'lidar': 'distance', 'distance': 'distance',
                     'lidaroccupancy': 'occupancy', 'occupancy': 'occupancy'}
BEST_MFREE_PERFORMANCES = {'austria': {'d4pg': 0.38, 'mpo': 0.36, 'ppo': 0.36, 'sac': 0.36, 'lstm-ppo': 0.36},
                           'columbia': {'d4pg': 2.06, 'mpo': 2.13, 'ppo': 2.09, 'sac': 1.97, 'lstm-ppo': 2.10},
                           'treitlstrassev2': {'d4pg': 0.77, 'mpo': 0.69, 'ppo': 0.66, 'sac': 0.30, 'lstm-ppo': 0.67}}
BEST_DREAMER_PERFORMANCES = {'austria': {'dreamer': 1.31},
                             'columbia': {'dreamer': 2.23},
                             'treitlstrassev2': {'dreamer': 2.00}}