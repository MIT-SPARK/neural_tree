"""
This file contains StanfordDataset class, which is used to load pre-processed room-object graphs from Stanford 3D Scene
Graph dataset (https://3dscenegraph.stanford.edu/).
"""
from neural_tree.h_tree import generate_jth, generate_node_labels
import torch
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
import networkx as nx
import pickle
import random


class StanfordDataset:
    def __init__(self, dataset_file_path):
        with open(dataset_file_path, 'rb') as input_file:
            dataset_dict, label_conversion_dict, num_labels = pickle.load(input_file)
        self.num_node_features = dataset_dict['x_list'][0].shape[1]
        self.num_classes = num_labels
        self.label_conversion_dict = label_conversion_dict
        self.name = 'Stanford_room_objects'
        self.__dataset = []
        for i in range(len(dataset_dict['x_list'])):
            self.__dataset.append(Data(x=torch.tensor(dataset_dict['x_list'][i]),
                                       y=torch.tensor(dataset_dict['y_list'][i]),
                                       edge_index=torch.tensor(dataset_dict['edge_index_list'][i]),
                                       object_mask=torch.tensor(dataset_dict['object_mask_list'][i]),
                                       room_mask=torch.tensor(dataset_dict['room_mask_list'][i])))
        self.add_label_mask()

    def __getitem__(self, item):
        return self.__dataset[item]

    def __len__(self):
        return len(self.__dataset)

    def add_label_mask(self):
        for data in self.__dataset:
            num_nodes = data.x.shape[0]
            data.y_room = torch.zeros(num_nodes, dtype=torch.bool)
            data.y_room[0] = True
            data.y_object = torch.ones(num_nodes, dtype=torch.bool)
            data.y_object[0] = False

    def get_diameter_list(self, weighted=True):
        """
        Get diameters of the junction tree hierarchies for each graph in the dataset.
        If weighted=True, the output list will contain copies of the same diameter for each node in the same graph.
        """
        diameter_list = []
        for data in self.__dataset:
            # Convert to networkx graph and then compute JTH
            G = pyg_utils.to_networkx(data, node_attrs=['x'])
            G = nx.to_undirected(G)
            G = generate_node_labels(G)
            G.graph = {'original': True}
            zero_feature = [0.0] * data.num_node_features
            G_jth, root_nodes = generate_jth(G, zero_feature=zero_feature)

            # save diameter to list
            diameter = nx.diameter(G_jth)
            if weighted:
                diameter_list += [diameter] * data.num_nodes
            else:
                diameter_list.append(diameter)
        return diameter_list

    def get_treewidth_upperbound_list(self, weighted=True):
        """
        Get upper bounds of treewidths for each graph in the dataset by using junction tree decomposition.
        If weighted=True, the output list will contain copies of the same bound for each node in the same graph.
        """
        tw_list = []
        for data in self.__dataset:
            # Convert to networkx graph and then compute JTH
            G = pyg_utils.to_networkx(data, node_attrs=['x'])
            G = nx.to_undirected(G)
            G = generate_node_labels(G)
            G.graph = {'original': True}
            zero_feature = [0.0] * data.num_node_features
            G_jth, root_nodes = generate_jth(G, zero_feature=zero_feature)

            # save diameter to list
            tw = max(len(G_jth.nodes[i]['clique_has']) for i in root_nodes) - 1
            if weighted:
                tw_list += [tw] * data.num_nodes
            else:
                tw_list.append(tw)
        return tw_list

    def generate_node_split(self, train_node_ratio, val_node_ratio, test_node_ratio=None):
        assert train_node_ratio > 0
        assert val_node_ratio >= 0
        if test_node_ratio is None:
            test_node_ratio = 1 - train_node_ratio - val_node_ratio
        assert test_node_ratio >= 0
        assert train_node_ratio + val_node_ratio + test_node_ratio <= 1

        num_nodes_total = 0
        node_locator_dict = dict()  # mapping global node_id in the dataset to specific data i and node index in data i
        for i, data in enumerate(self.__dataset):
            num_nodes = data.x.shape[0]
            node_locator_dict.update(dict(zip(list(range(num_nodes_total, num_nodes_total + num_nodes)),
                                              [(i, node_id) for node_id in range(num_nodes)])))
            num_nodes_total += num_nodes
            # initialize masks (starting from all test node)
            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask = torch.ones(num_nodes, dtype=torch.bool)

        num_train_nodes = int(num_nodes_total * train_node_ratio)
        num_val_nodes = int(num_nodes_total * val_node_ratio)
        if train_node_ratio + val_node_ratio + test_node_ratio == 1:
            num_unused_nodes = 0
        else:
            num_unused_nodes = num_nodes_total - num_train_nodes - num_val_nodes - \
                               int(num_nodes_total * test_node_ratio)

        # convert test node to train node
        train_nodes_found = 0
        while train_nodes_found < num_train_nodes:
            data_id, node_id = node_locator_dict[random.randint(0, num_nodes_total - 1)]
            if self.__dataset[data_id].test_mask[node_id].item() is True:
                self.__dataset[data_id].train_mask[node_id] = True
                self.__dataset[data_id].test_mask[node_id] = False
                train_nodes_found += 1

        # convert test node to val node
        val_nodes_found = 0
        while val_nodes_found < num_val_nodes:
            data_id, node_id = node_locator_dict[random.randint(0, num_nodes_total - 1)]
            if self.__dataset[data_id].test_mask[node_id].item() is True:
                self.__dataset[data_id].val_mask[node_id] = True
                self.__dataset[data_id].test_mask[node_id] = False
                val_nodes_found += 1

        # if train/val/test ratios do not sum up to 1 (i.e. num_unused_nodes > 0), remove some test nodes
        unused_nodes_found = 0
        while unused_nodes_found < num_unused_nodes:
            data_id, node_id = node_locator_dict[random.randint(0, num_nodes_total - 1)]
            if self.__dataset[data_id].test_mask[node_id].item() is True:
                self.__dataset[data_id].test_mask[node_id] = False
                unused_nodes_found += 1

        print('Train/val/test nodes split: {}/{}/{}.'.
              format(num_train_nodes, num_val_nodes,
                     num_nodes_total - num_train_nodes - num_val_nodes - num_unused_nodes))
