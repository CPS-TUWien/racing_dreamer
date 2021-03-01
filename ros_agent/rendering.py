import numpy as np

import cmocean
import networkx as nx
import matplotlib.pyplot as plt
plt.ion()

class DynamicNormalization:
    def __init__(self, target_min=0, target_max=1, shape=(1, 1), globally=True):
        self._true_min = np.ones(shape) * np.Inf
        self._true_max = np.ones(shape) * -np.Inf
        self._target_min = target_min
        self._target_max = target_max
        self._globally = globally
        self._shape = shape

    def __call__(self, array):
        if self._globally:
            self._true_min = np.min(np.concatenate([array, self._true_min], axis=1)).reshape(self._shape)
            self._true_max = np.max(np.concatenate([array, self._true_max], axis=1)).reshape(self._shape)
        else:
            self._true_min = np.min(np.concatenate([array, self._true_min]), axis=0).reshape(self._shape)
            self._true_max = np.max(np.concatenate([array, self._true_max]), axis=0).reshape(self._shape)
        array = np.where(self._true_max - self._true_min > 0.001, (array - self._true_min) / (self._true_max - self._true_min), 0)  # normalize in 0,1
        array = (self._target_max - self._target_min) * array + self._target_min
        return array


class RenderLTC:
    """ Class to manage the rendering of LTC cell activity."""
    _layers = {'sensor': 0, 'inter': 1, 'command': 2, 'motor': 3}
    # layout: range within spread neurons over each layer
    _desired_miny = {'sensor': -1, 'inter': -1, 'command': -0.5, 'motor': -0.2}
    _desired_maxy = {'sensor': +1, 'inter': +1, 'command': +0.5, 'motor': +0.2}
    # rendering stuff, for now manage only `sensor` and `non_sensor` (ie. inter, cmd, motor neurons look the same)
    _node_sizes = {'sensor': 100, 'non_sensor': 100}
    _node_shapes = {'sensor': 's', 'non_sensor': 'o'}
    _node_cmaps = {'sensor': cmocean.cm.thermal, 'non_sensor': cmocean.cm.thermal}


    def __init__(self, wirings):
        self._wirings = wirings
        self._adjacency_matrix = wirings.adjacency_matrix
        self._sensory_adjacency_matrix = wirings.sensory_adjacency_matrix
        self._G, self._node_lists, self._edge_lists, self._sensor_node_map, self._neuron_node_map = self._build_graph()
        self._subgraphs = {n_type: self._G.subgraph(self._node_lists[n_type]) for n_type in self._node_lists}
        self._pos = self._build_graph_layout()
        self._norm_embeds = DynamicNormalization(target_min=-1, target_max=+1, shape=(1, self._sensory_adjacency_matrix.shape[0]), globally=False)
        self._norm_states = DynamicNormalization(target_min=-1, target_max=+1, globally=True)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        self._already_render = False
        self._net_ax = plt.gca()
        divider = make_axes_locatable(self._net_ax)
        self._cbar_ax = divider.append_axes("bottom", size="5%", pad="2%")

    def render(self, embed, state):
        canvas = self._net_ax.figure.canvas
        embed = self._norm_embeds(embed)
        state = self._norm_states(state)
        ns = nx.draw_networkx_nodes(self._subgraphs['sensor'], pos=self._pos, node_size=self._node_sizes['sensor'],
                                    node_shape=self._node_shapes['sensor'], node_color=embed, edgecolors='k',
                                    cmap=self._node_cmaps['sensor'], vmin=0, vmax=+1, ax=self._net_ax)
        no = nx.draw_networkx_nodes(self._subgraphs['non_sensor'], pos=self._pos,
                                    node_size=self._node_sizes['non_sensor'],
                                    node_shape=self._node_shapes['non_sensor'], node_color=state, edgecolors='k',
                                    cmap=self._node_cmaps['non_sensor'], vmin=-1, vmax=+1, ax=self._net_ax)
        if not self._already_render:
            nx.draw_networkx_edges(self._G, pos=self._pos, edgelist=self._edge_lists['activation'], arrowstyle='-|>',
                                   edge_color='black', alpha=0.7, ax=self._net_ax)
            nx.draw_networkx_edges(self._G, pos=self._pos, edgelist=self._edge_lists['inibition'], arrowstyle='-[',
                                   arrowsize=2, edge_color='black', alpha=0.7, ax=self._net_ax)
            plt.colorbar(no, orientation='horizontal', cax=self._cbar_ax)
            self._already_render = True
        canvas.draw()
        plt.pause(0.0001)
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        image = data.reshape(canvas.get_width_height()[::-1] + (3,))
        return image

    def _build_graph(self):
        """ Starting from adj matrices, build the graph and the needed datastructures.
            Returns:
                G: networkx graph
                nodelists: dict of lists of node ids. ie. 'neuron_type' -> [node_id]
                edgelists: dict of lists of edge ids. ie. 'edge_type' -> [edge_id]
                sensor_node_map: map sensor_id to node_id
                neuron_node_map: map neuron_id to node_id
        """
        G = nx.DiGraph()  # directed graph
        next_node_id = 0
        sensor_node_map, neuron_node_map = {}, {}
        nodelists = {'sensor': [], 'inter': [], 'command': [], 'motor': [], 'non_sensor': []}
        # add neurons
        for sensor_node in range(self._sensory_adjacency_matrix.shape[0]):
            n_type = 'sensor'
            G.add_node(sensor_node, layer=self._layers[n_type])
            nodelists[n_type].append(next_node_id)
            sensor_node_map[sensor_node] = next_node_id  # map `sensor_id` to `node_id` in the graph
            next_node_id += 1
        for neuron_node in range(self._adjacency_matrix.shape[0]):
            n_type = self._wirings.get_type_of_neuron(neuron_node)
            G.add_node(next_node_id, layer=self._layers[n_type])
            nodelists[n_type].append(next_node_id)
            nodelists['non_sensor'].append(next_node_id)
            neuron_node_map[neuron_node] = next_node_id  # map `neuron_id` to `node_id` in the graph
            next_node_id += 1
        # add edges
        edgelists = {'activation': [], 'inibition': []}
        for sensor_node in range(self._sensory_adjacency_matrix.shape[0]):
            for neuron_node in np.nonzero(self._sensory_adjacency_matrix[sensor_node, :])[0]:
                source = sensor_node_map[sensor_node]
                target = neuron_node_map[neuron_node]
                G.add_edge(source, target)
                if self._sensory_adjacency_matrix[sensor_node, neuron_node] > 0:  # already filtered `nonzero`, >0 means +1
                    edgelists['activation'].append((source, target))
                else:
                    edgelists['inibition'].append((source, target))  # <0 means -1
        for first_neuron in range(self._adjacency_matrix.shape[0]):
            for second_neuron in np.nonzero(self._adjacency_matrix[first_neuron, :])[0]:
                source = neuron_node_map[first_neuron]  # map to node ids
                target = neuron_node_map[second_neuron]
                G.add_edge(source, target)
                if self._adjacency_matrix[first_neuron, second_neuron] > 0:  # already filtered `nonzero`, >0 means +1
                    edgelists['activation'].append((source, target))
                else:
                    edgelists['inibition'].append((source, target))  # <0 means -1
        return G, nodelists, edgelists, sensor_node_map, neuron_node_map

    def _build_graph_layout(self):
        """ Create multi-layer structure according to the `layer` field in node definition.
            Returns:
                pos:    networkx positioning as dict `node_id`->Tuple(x, y)
        """
        pos = nx.multipartite_layout(self._G, subset_key="layer")
        original_miny = {'sensor': min([y for x, y in [pos[node] for node in self._node_lists['sensor']]]),
                         'inter': min([y for x, y in [pos[node] for node in self._node_lists['inter']]]),
                         'command': min([y for x, y in [pos[node] for node in self._node_lists['command']]]),
                         'motor': min([y for x, y in [pos[node] for node in self._node_lists['motor']]])}
        original_maxy = {'sensor': max([y for x, y in [pos[node] for node in self._node_lists['sensor']]]),
                         'inter': max([y for x, y in [pos[node] for node in self._node_lists['inter']]]),
                         'command': max([y for x, y in [pos[node] for node in self._node_lists['command']]]),
                         'motor': max([y for x, y in [pos[node] for node in self._node_lists['motor']]])}
        for node, (x, y) in pos.items():
            for n_type in ['sensor', 'inter', 'command', 'motor']:
                if node in self._node_lists[n_type]:
                    newy = (self._desired_maxy[n_type] - self._desired_miny[n_type]) * (y - original_miny[n_type]) / (
                            original_maxy[n_type] - original_miny[n_type]) + self._desired_miny[n_type]
                    pos[node] = np.array([x, newy])
        return pos


if __name__ == "__main__":
    pass
