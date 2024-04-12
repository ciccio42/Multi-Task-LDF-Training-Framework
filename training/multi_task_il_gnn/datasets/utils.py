import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


OBJECTS_POS_DIM = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox', 'bin'],
        'bin_position': [0.18, 0.00, 0.75],
        'obj_dim': {'greenbox': [0.05, 0.055, 0.045],  # W, H, D
                    'yellowbox': [0.05, 0.055, 0.045],
                    'bluebox': [0.05, 0.055, 0.045],
                    'redbox': [0.05, 0.055, 0.045],
                    'bin': [0.6, 0.06, 0.15],
                    'single_bin': [0.15, 0.06, 0.15]}
    }
}

NUM_OBJ_NUM_TARGET_PER_OBJ = {'pick_place': (4, 4),
                              'nut_assembly': (3, 3)}

ENV_INFORMATION = {
    'camera_names': {'camera_front', 'camera_lateral_right', 'camera_lateral_left'},
    'camera_pos':
        {
            'camera_front': [[0.45, -0.002826249197217832, 1.27]],
            'camera_lateral_left': [[-0.32693157973832665, 0.4625646268626449, 1.3]],
            'camera_lateral_right': [[-0.3582777207605626, -0.44377700364575223, 1.3]],
    },
    'camera_orientation':
        {
            'camera_front':  [0.6620018964346217, 0.26169506249574287, 0.25790267731943883, 0.6532651777140575],
            'camera_lateral_left': [-0.3050297127346233,  -0.11930536839029657, 0.3326804927221449, 0.884334095907446],
            'camera_lateral_right': [0.860369883903888, 0.3565444300005689, -0.1251454368177692, -0.3396500627826067],
    },
    'camera_fovy': 60,
    'img_dim': [200, 360]
}


def plot_graph(data, save_path=None):
    # Extracting node features and edge indices
    x = data.x.numpy()
    edge_index = data.edge_index.numpy()

    # Creating a directed graph
    G = nx.DiGraph()

    # Adding nodes with features
    for i in range(x.shape[0]):
        G.add_node(i, feature=x[i])

    # Adding edges
    for src, dst in edge_index.T:
        G.add_edge(src, dst)

    # Getting the last feature values
    last_features = x[:, -1]

    # Plotting the graph
    pos = nx.spring_layout(G)  # positions for all nodes
    node_colors = ['blue' if feature ==
                   1 else 'red' for feature in last_features]
    nx.draw(G, pos, node_color=node_colors, node_size=300, with_labels=True, labels={i: str(i) for i in G.nodes()},
            linewidths=0.5, font_size=8)

    # Save the plot if save_path is provided
    plt.savefig("graph_debug.png")


def compute_object_features(task_name: str, object_name: str):
    if task_name == "pick_place":
        if object_name == "greenbox":
            return np.array([29, 122, 41])
        elif object_name == "yellowbox":
            return np.array([255, 250, 160])
        elif object_name == "bluebox":
            return np.array([9, 51, 93])
        elif object_name == "redbox":
            return np.array([229, 0, 20])
        elif object_name == 'bin':
            return np.array([168, 116, 76])


class Graph():
    def __init__(self, feat_vect: np.array, edge_indx: np.array):
        self._feat_vect = feat_vect
        self._edge_indx = edge_indx
