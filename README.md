## Requirements
- python >= 3.7
- CUDA >= 9.2 (for PyTorch Geometric)

## Dependencies
### PyTorch and PyTorch Geometric
The graph neural network architecture in the repository is built using [PyTorch](https://pytorch.org/get-started/locally/) 
and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html). 
Follow the official installation guide to install this library. 
An easy way to install PyTorch and PyTorch geometric is to use pip wheels. 
Below is an example installation code (for CUDA 10.1):
```
pip install scipy Cython
pip install torch==1.6.0 torchvision==0.7.0
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-geometric
```
We have tested this code repo on a Ubuntu 18.04 machine with PyTorch versions 1.6.0, 1.8.0, and PyTorch Geometric 2.0.1 with CUDA 10.1.
Check this [link](https://pytorch.org/get-started/previous-versions/) for different PyTorch installation options and this [link](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) for PyTorch Geometric.
Note that our preprocessed scene graph dataset might not work with other PyTorch Geometric versions.

### Other dependencies
We used [TensorboardX](https://tensorboardx.readthedocs.io/en/latest/index.html) for visualizing training progress, 
[NetworkX](https://networkx.org/) for constructing (junction) tree decomposition, pickle for saving and loading preprocessed
datasets.
```
pip install tensorboardX
pip install networkx
```

## Citations
```
@inproceedings{NEURIPS2021_NeuralTree,
 author = {Talak, Rajat and Hu, Siyi and Peng, Lisa and Carlone, Luca},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Neural Trees for Learning on Graphs},
 year = {2021}
}
```

## Dataset
We tested our implementation on the room-object scene graphs, modified from the verified tiny split in [Stanford 3D Scene Graph](https://3dscenegraph.stanford.edu/) dataset.
The processed dataset file and the original dataset license is in the `data` folder.

We also performed test on the Planetoid citation datasets (Cora, Citeseer, Pubmed). These datasets are directly obtained through PyTorch Geometric. 

See python scripts in `experiment` to reproduce our experiments on these datasets.

## Quick Start
```
python setup.py develop
python scene_graph_experiment.py
```

## Implementation note
This repo contains code that run the original architectures (GCN, GAT, GraphSAGE, GIN) and the neural tree architecture 
using the same aggregation functions on scene graphs and citation networks. 
You can set `algorithm` to either `original` or `neural-tree` as shown in the example code, `scene_graph_experiment.py`.

Our proposed `neural-tree` architecture runs message passing on a hierarchical tree data structure, called "H-tree".
An H-tree can be computed via any tree decomposition.
We implemented junction tree decomposition for our experiment using NetworkX and we call it "junction tree hierarchy".
See `generate_jth` and `sample_and_generate_jth` (with bounded treewidth subsampling) in 
`neural_tree/h_tree/generate_junction_tree_hierarchies.py`.

To train neural tree algorithm on H-tree for node classification, our current implementation generates an H-tree for 
each (train, validation, test) node to be classified. For scene graphs, where the graph is small, the same H-tree is 
used for all nodes on the same graph, with different leaf node mask for aggregating final hidden states. 
Note that some parts of the code assume input graph contains `room_mask` and `object_mask`, and hence are specific to 
scene graphs. You will need to modify these for other applications (see `neural_tree/h_tree/h_tree_utils.py` and 
`neural_tree/utils/base_training_job.py`).
For citation network, where the graph is large, an H-tree is constructed for each connected component of the dataset 
after bounded treewidth subsampling (run `experiment/preprocess_citation_dataset_with_subsampling.py` after specifying 
dataset name and treewidth bound). 
Then, a sub-tree is segmented for each node to be classified. 
If there are T number of iterations, the sub-tree includes all leaf nodes corresponding to the classification node, and
their neighbors that are within T-hops, such that message passing on the sub-trees is equivalent to that on the full H-tree.

### Hyper-parameters
The hyper-parameters are broken down to `network_params`, `optimization_params`, `dataset_params`, and 
`neural_tree_params` (for neural tree only). See example below:
```
network_params = {'conv_block': 'GCN',
                  'hidden_dim': 128,
                  'num_layers': 4,
                  'GAT_hidden_dims': [8, 8],
                  'GAT_heads': [8, 1],
                  'GAT_concats': [True, False],
                  'dropout': 0.25}
optimization_params = {'lr': 0.01,
                       'num_epochs': 1000,
                       'weight_decay': 0.0}
dataset_params = {'batch_size': 128,
                  'shuffle': True}
neural_tree_params = {'min_diameter': 1,
                      'max_diameter': None,
                      'sub_graph_radius': None}
```
`hidden_dim` and `num_layers` are for `GCN`, `GraphSAGE` and `GIN` aggregation functions. 
If all nodes are homogeneous, i.e. in the same label space (e.g. citation networks), the output dimension of the last 
convolution layer will be the same as the number of labels.
If nodes are in different label space (e.g. scene graphs), the output dimension of the last convolution layer will be 
the specified `hidden_dim`, same as other convolution layers. A post message passing MLP (implemented as a 1-layer MLP) 
is used after last convolution to map the outputs to desired labels spaces.

`GAT_hidden_dims`, `GAT_heads`, `GAT_concats` are for `GAT` aggregation function so that each GAT convolution layer has 
its own hyper-parameter, and `hidden_dim` and `num_layers` will be ignored. 
The number of convolution layers is the same as the length of `GAT_heads` and `GAT_concats`.
In the case of homogenous nodes, the last layer hidden dimension does not need to be specified and hence the length of 
`GAT_hidden_dims` will be 1 less than the other two hyper-parameters; otherwise the lengths should all be the same.

The `optimization_params` contains hyper-parameters for the Adam optimizer and `dataset_params` are related to 
PyTorch Geometric dataset loader.

The diameter hyper-parameters in `neural_tree_params` will filter out some H-tree and hence might reduce the number of 
classification nodes. Do not change this unless this is a desired modification. 
The `sub_graph_radius` will segment the original graph to an ego graph (`None` means the entire graph), where the center 
is the node to be classified and radius is bounded by the specified hyper-parameter, when constructing H-trees. 
This can speed up H-tree generation for large graphs. However, we recommend using treewidth subsampling and use the 
entire graph when constructing H-tree to avoid loosing global information of the data.
