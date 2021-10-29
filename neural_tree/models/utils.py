import torch.nn as nn
import torch_geometric.nn as pyg_nn


def build_conv_layer(conv_block, input_dim, hidden_dim):
    """
    Build a PyTorch Geometric convolution layer given specified input and output dimension.
    """
    if conv_block == 'GCN':
        return pyg_nn.GCNConv(input_dim, hidden_dim)
    elif conv_block == 'GIN':
        return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim)), eps=0., train_eps=True)
    elif conv_block == 'GraphSAGE':
        return pyg_nn.SAGEConv(input_dim, hidden_dim, normalize=False, bias=True)
    else:
        return NotImplemented


def build_GAT_conv_layers(input_dim, hidden_dims, heads, concats, dropout=0.):
    """
    Build a list of PyTorch Geometric GAT convolution layers given input dimension and dropout ratio. This function also
     requires hidden dimensions, number of attention heads, concatenation flags for all layers.
    """
    assert len(hidden_dims) == len(heads)
    assert len(hidden_dims) == len(concats)
    convs = nn.ModuleList()
    convs.append(pyg_nn.GATConv(input_dim, hidden_dims[0], heads=heads[0], concat=concats[0], dropout=dropout))
    for i in range(1, len(hidden_dims)):
        if concats[i - 1]:
            convs.append(pyg_nn.GATConv(hidden_dims[i - 1] * heads[i - 1], hidden_dims[i], heads=heads[i],
                                        concat=concats[i], dropout=dropout))
        else:
            convs.append(pyg_nn.GATConv(hidden_dims[i - 1], hidden_dims[i], heads=heads[i], concat=concats[i],
                                        dropout=dropout))
    return convs
