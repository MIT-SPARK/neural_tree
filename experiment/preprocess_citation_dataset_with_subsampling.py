"""
This script construct H-trees for the Planetoid citation datasets (cora, citeseer, pubmed). The result trees, together
with the original dataset, will be saved to preprocessed_<dataset_name>/<dataset_name>_tw<tree_width>.pkl.
"""

from torch_geometric.datasets import Planetoid
import torch_geometric.utils as pyg_utils
from neural_tree.h_tree import sample_and_generate_jth
import pickle
import time
from os import path, mkdir
import sys
import networkx as nx

# run control parameters
treewidth = 1
need_root_tree = True
remove_edges_every_layer = True
dataset_name = 'cora'   # 'cora', 'citeseer', 'pubmed'

# output file
output_dir = path.dirname(path.abspath(__file__)) + '/preprocessed_' + dataset_name
tree_output_file_path = path.join(output_dir, '{}_tw{}.pkl'.format(dataset_name, treewidth))


if __name__ == '__main__':
    # check existing file/folder
    if path.exists(tree_output_file_path):  # tree decomposition already computed
        print('Tree decomposition file already found in:', tree_output_file_path)
        sys.exit(0)
    if path.exists(output_dir):
        print('Found output directory:', output_dir)
    else:
        mkdir(output_dir)

    tic = time.perf_counter()

    # load dataset
    dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
    if len(dataset) > 1:
        raise RuntimeError('Input dataset should not contain more than one data object.')

    # subsample each connected component in the dataset and then convert it to JTH
    data = dataset[0]
    G_all = pyg_utils.to_networkx(data, node_attrs=['x'], to_undirected=True)
    G_list = [G_all.subgraph(c).copy() for c in nx.connected_components(G_all)]
    print('Found {} connected component in the dataset.'.format(len(G_list)))
    zero_feature = [0.0] * dataset.num_node_features
    G_jth_list = []
    for G in G_list:
        G_sampled, G_jth, root_nodes = sample_and_generate_jth(G, k=treewidth, zero_feature=zero_feature,
                                                               copy_node_attributes=['x'],
                                                               need_root_tree=need_root_tree,
                                                               remove_edges_every_layer=remove_edges_every_layer,
                                                               verbose=True)
        G_jth_list.append(G_jth)

    with open(tree_output_file_path, 'wb') as output_file:
        pickle.dump({'dataset': dataset, 'tree_decomposition': G_jth_list}, output_file, pickle.HIGHEST_PROTOCOL)

    toc = time.perf_counter()
    print('Saving sampled tree to: {} (time elapsed: {:.3f} min).'.format(tree_output_file_path, (toc - tic) / 60))
