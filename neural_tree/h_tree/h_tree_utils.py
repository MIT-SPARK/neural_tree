import networkx as nx
import torch
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
from copy import deepcopy
from neural_tree.h_tree import generate_jth, generate_node_labels


class HTreeDataset:
    """
    H-Tree dataset
        data_list: a list of torch_geometric.data.Data instances (graph classification) or a list of three such lists
         corresponding to train, val, test split (node classification)
        num_node_features:  int
        num_classes:        int
        name:               string
        task:               string (node or graph)
    """
    def __init__(self, data_list, num_node_features, num_classes, name, task, data_list_original=None):
        assert isinstance(num_node_features, int)
        if data_list_original is not None:
            assert isinstance(data_list_original[0], Data)
        assert isinstance(num_classes, int) or isinstance(num_classes, tuple)
        assert task == 'graph' or task == 'node'
        if task == 'graph':
            assert isinstance(data_list[0], Data)
        else:
            assert len(data_list) == 3
            for i in range(3):
                assert len(data_list) == 0 or isinstance(data_list[i][0], Data)

        self.dataset_jth = data_list
        self.dataset_original = data_list_original
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.name = name
        self.task = task


def convert_to_networkx_jth(data: Data, task='graph', node_id=None, radius=None):
    """
    Convert a graph or its subgraph given id and radius (an ego graph) to junction tree hierarchy. The node features in
    the input graph will be copied to the corresponding leaf nodes of the output tree decomposition.
    :param data: torch_geometric.data.Data, input graph
    :param task: 'node' or 'graph'
    :param node_id: int, node id in the input graph to be classified (only used when task='node')
    :param radius: int, radius of the ego graph around classification node to be converted (only used when task='node')
    :returns: data_jth, G_jth, root_nodes
    """
    # Convert to networkx graph
    G = pyg_utils.to_networkx(data, node_attrs=['x'])
    G = nx.to_undirected(G)

    if task == 'graph':
        if nx.is_connected(G) is False:
            raise RuntimeError('[Input graph] is disconnected.')

    else:  # task == 'node'
        if radius is not None:
            G_subgraph = nx.ego_graph(G, node_id, radius=radius, undirected=False)
            extracted_id = [i for i in G_subgraph.nodes.keys()]
            G_subgraph = nx.relabel_nodes(G_subgraph, dict(zip(extracted_id, list(range(len(G_subgraph))))), copy=True)
            G = generate_node_labels(G_subgraph)
        else:
            extracted_id = [i for i in G.nodes.keys()]
            G = generate_node_labels(G)
        # index of the classification node in the extracted graph, for computing leaf_mask
        classification_node_id = extracted_id.index(node_id)

    is_clique_graph = True if len(list(G.edges)) == G.number_of_nodes() * (G.number_of_nodes() - 1) / 2 else False

    # Create junction tree hierarchy
    G.graph = {'original': True}
    zero_feature = [0.0] * data.num_node_features
    G_jth, root_nodes = generate_jth(G, zero_feature=zero_feature)

    # Convert back to torch Data (change first clique_has value to avoid TypeError when calling pyg_utils.from_networkx
    if is_clique_graph:  # clique graph
        G_jth.nodes[0]['clique_has'] = 0
    else:
        G_jth.nodes[0]['clique_has'] = [0]
    data_jth = pyg_utils.from_networkx(G_jth)

    try:
        data_jth['diameter'] = nx.diameter(G_jth)
    except nx.NetworkXError:
        data_jth['diameter'] = 0
        print('junction tree hierarchy disconnected.')
        return data_jth

    if task == 'node':
        data_jth['classification_node'] = classification_node_id

    return data_jth, G_jth, root_nodes


def convert_room_object_graph_to_jth(data: Data, node_id=None, radius=None) -> Data:
    """
    Convert a room-object graph or its subgraph given id and radius (an ego graph) to junction tree hierarchy with
     leaf_mask attribute in addition to input feature x and label y. data_jth.leaf_mask is a BoolTensor of dimension
     [data_jth.num_nodes] specifying leaf nodes.
    :param data: torch_geometric.data.Data, input graph
    :param node_id: int, node id in the input graph to be classified (only used when task='node')
    :param radius: int, radius of the ego graph around classification node to be converted (only used when task='node')
    :returns: data_jth
    """
    data_jth, G_jth, root_nodes = convert_to_networkx_jth(data, 'node', node_id, radius)

    # Save leaf_mask
    leaf_mask = torch.zeros(data_jth.num_nodes, dtype=torch.bool)
    for v, attr in G_jth.nodes('type'):
        if attr == 'node' and G_jth.nodes[v]['clique_has'] == data_jth['classification_node']:
            leaf_mask[v] = True
    data_jth['leaf_mask'] = leaf_mask
    data_jth.y = data.y[node_id]
    data_jth.y_room = data.room_mask[node_id]
    data_jth.y_object = data.object_mask[node_id]
    assert data_jth.y_room.item() != data_jth.y_object.item()

    data_jth.clique_has = None
    data_jth.type = None
    data_jth.classification_node = None
    return data_jth


def convert_room_object_graph_to_same_jths(data: Data, min_diameter=None, max_diameter=None):
    """
    Convert a room-object graph to the same junction tree hierarchy for all the original nodes.
    This function outputs three lists of jth's corresponding to original nodes in train_mask, val_mask and tes_mask.
    :param data: torch_geometric.data.Data, input graph
    :param min_diameter: int, minimum diameter of the results jth, below which this function returns three empty lists
    :param max_diameter: int, maximum diameter of the results jth, below which this function returns three empty lists
    :returns: train_list, val_list, test_list
    """
    assert isinstance(data, Data)
    train_list = []
    val_list = []
    test_list = []

    data_jth, G_jth, root_nodes = convert_to_networkx_jth(data, 'node', 0, None)

    # return empty lists if diameter is beyond specified bound
    try:
        data_jth['diameter'] = nx.diameter(G_jth)
    except nx.NetworkXError:
        data_jth['diameter'] = 0
        print('junction tree hierarchy disconnected.')
        return data_jth
    if (min_diameter is not None and data_jth.diameter < min_diameter) or \
            (max_diameter is not None and data_jth.diameter > max_diameter):
        return train_list, val_list, test_list

    # prepare modified copies of data_jth based on train/val/test masks
    data_jth.clique_has = None
    data_jth.type = None
    data_jth.classification_node = None
    for node_id in range(data.num_nodes):
        if data.train_mask[node_id].item() or data.val_mask[node_id].item() or data.test_mask[node_id].item():
            # create a copy of data_jth and related attributes
            data_jth_i = deepcopy(data_jth)
            leaf_mask = torch.zeros(data_jth.num_nodes, dtype=torch.bool)
            for v, attr in G_jth.nodes('type'):
                if attr == 'node' and G_jth.nodes[v]['clique_has'] == node_id:
                    leaf_mask[v] = True
            data_jth_i['leaf_mask'] = leaf_mask
            data_jth_i.y = data.y[node_id]
            data_jth_i.y_room = data.room_mask[node_id]
            data_jth_i.y_object = data.object_mask[node_id]
            assert data_jth_i.y_room.item() != data_jth_i.y_object.item()
            # save to lists
            if data.train_mask[node_id].item() is True:
                train_list.append(data_jth_i)
            if data.val_mask[node_id].item() is True:
                val_list.append(data_jth_i)
            if data.test_mask[node_id].item() is True:
                test_list.append(data_jth_i)
    return train_list, val_list, test_list


def convert_dataset_to_junction_tree_hierarchy(dataset, task, min_diameter=None, max_diameter=None, radius=None):
    """
    Convert a torch.dataset object or a list of torch.Data to a junction tree hierarchies.
    :param dataset:     a iterable collection of torch.Data objects
    :param task:            str, 'graph' or 'node'
    :param min_diameter:    int
    :param max_diameter:    int
    :param radius:          int, maximum radius of extracted sub-graphs for node classification
    :return: if task = 'graph', return a list of torch.Data objects in the same order as in dataset;
     else (task = 'node'), return a list of three such lists, for nodes and the corresponding subgraph in train_mask,
     val_mask, and test_mask respectively.
    """
    if task == 'graph':
        raise RuntimeError('Junction tree hierarchy not implemented for graph classification -- work in progress.')
    elif task == 'node':
        train_list = []
        val_list = []
        test_list = []
        for data in dataset:
            if radius is None:  # for nodes in the same graph, use the same junction tree hierarchy
                train_graphs, val_graphs, test_graphs \
                    = convert_room_object_graph_to_same_jths(data, min_diameter, max_diameter)
                train_list += train_graphs
                val_list += val_graphs
                test_list += test_graphs
            else:               # otherwise, create jth for each node separately
                for i in range(data.num_nodes):
                    if data.train_mask[i].item() or data.val_mask[i].item() or data.test_mask[i].item():
                        data_jth = convert_room_object_graph_to_jth(data, node_id=i, radius=radius)
                        if (min_diameter is None or data_jth.diameter >= min_diameter) and \
                                (max_diameter is None or data_jth.diameter <= max_diameter):
                            if data.train_mask[i].item() is True:
                                train_list.append(data_jth)
                            elif data.val_mask[i].item() is True:
                                val_list.append(data_jth)
                            elif data.test_mask[i].item() is True:
                                test_list.append(data_jth)
        return [train_list, val_list, test_list]
    else:
        raise Exception("must specify if task is 'graph' or 'node' classification")


def get_subtrees_from_htree(data, G_htree, radius):
    """
    Segment sub-trees from input H-tree such that each sub-tree corresponds to a label node in the original graph.
    Note: if the original graph is disconnected, input H-tree should be computed from one of the connected component of
    the original graph.
    :param data: torch_geometric.data.Data, input graph
    :param G_htree: nx.Graph, H-tree decomposition of the (subsampled) original graph or one of the connected
    (subsampled) component of the original graph
    :param radius: int, furthest neighbor node from leaf nodes corresponding to a label node
    :return: train_list, val_list, test_list
    """
    # save leaf node indices for each node in the original graph
    leaf_nodes_list = [None] * data.num_nodes
    original_idx_set = set()      # indices of original nodes in data that are in G_jth
    for i, attr in G_htree.nodes('type'):
        if attr == 'node':
            original_idx = G_htree.nodes[i]['clique_has']
            original_idx_set.add(original_idx)
            if leaf_nodes_list[original_idx] is None:
                leaf_nodes_list[original_idx] = [i]
            else:
                leaf_nodes_list[original_idx].append(i)
    original_idx_list = list(original_idx_set)
    num_original_nodes = len(original_idx_list)

    # loop through each node in the original graph
    train_list = []
    val_list = []
    test_list = []
    data_mask = data.train_mask + data.val_mask + data.test_mask  # classification nodes
    progress_threshold = 0
    max_num_nodes = 0
    for j in range(num_original_nodes):
        original_idx = original_idx_list[j]
        if 100.0 * (j + 1) / num_original_nodes > progress_threshold + 10:
            progress_threshold += 10
        if data_mask[original_idx].item() is True:
            # segment subtree from the complete jth using specified radius from leaf nodes
            leaf_nodes = leaf_nodes_list[original_idx]
            G_subtree = nx.ego_graph(G_htree, leaf_nodes[0], radius=radius, undirected=False)
            for leaf_node in leaf_nodes[1:]:  # add other subtrees if there are multiple leaf nodes
                if G_subtree.number_of_nodes() == G_htree.number_of_nodes():
                    break
                H_subtree = nx.ego_graph(G_htree, leaf_node, radius=radius, undirected=False)
                G_subtree = nx.compose(G_subtree, H_subtree)
            extracted_id = [i for i in G_subtree.nodes.keys()]
            G_subtree = nx.relabel_nodes(G_subtree, dict(zip(extracted_id, list(range(len(G_subtree))))), copy=True)

            # convert G_subtree to torch data
            leaf_mask = torch.zeros(G_subtree.number_of_nodes(), dtype=torch.bool)
            for v, attr in G_subtree.nodes('type'):
                if attr == 'node' and G_subtree.nodes[v]['clique_has'] == original_idx:
                    leaf_mask[v] = True
                del G_subtree.nodes[v]['clique_has']
                del G_subtree.nodes[v]['type']
            data_subtree = pyg_utils.from_networkx(G_subtree)
            data_subtree.leaf_mask = leaf_mask
            data_subtree.y = data.y[original_idx]
            if nx.is_connected(G_subtree):
                data_subtree.diameter = nx.diameter(G_subtree)
            else:
                data_subtree.diameter = 0
            if data_subtree.num_nodes > max_num_nodes:
                max_num_nodes = data_subtree.num_nodes

            # save subtree
            if data.train_mask[original_idx].item() is True:
                train_list.append(data_subtree)
            elif data.val_mask[original_idx].item() is True:
                val_list.append(data_subtree)
            else:
                test_list.append(data_subtree)

    return train_list, val_list, test_list
