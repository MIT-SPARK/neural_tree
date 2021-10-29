from neural_tree.models.utils import build_conv_layer, build_GAT_conv_layers
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm.batch_norm import BatchNorm


class BasicNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, task='node', conv_block='GCN', hidden_dim=None, num_layers=None,
                 GAT_hidden_dims=None, GAT_heads=None, GAT_concats=None, dropout=0.25):
        """
        This BasicNetwork class implements basic message passing on graphs with different aggregation functions.
        If output_dim is a tuple, the last message passing iteration outputs final hidden state of size hidden_dim and
        an additional linear layer (post message passing operation) is used to generate output of different dimensions;
        otherwise, the last message passing iteration outputs final hidden state of size output_dim.
        :param input_dim: int, input feature dimension
        :param output_dim: int or a tuple of int, output dimension(s), i.e. the number of labels
        :param task: string, only 'node' classification is implemented now
        :param hidden_dim: int, the output dimension of the graph convolution operations (ignored if conv_block='GAT')
        :param num_layers: int, the number of graph convolution iterations (ignored if conv_block='GAT')
        :param GAT_hidden_dims: list of int, the output dimensions of GAT convolution operations (ignored if
        conv_block!='GAT')
        :param GAT_heads: list of int, the number of attention heads of GAT convolution operation (ignored if
        conv_block!='GAT')
        :param GAT_concats GAT_heads: list of bool, concatenate output of multi-head GAT convolution operation (ignored
        if conv_block!='GAT')
        :param dropout: float, dropout ratio during training
        """
        super(BasicNetwork, self).__init__()
        self.task = task
        self.conv_block = conv_block
        self.num_layers = num_layers if conv_block != 'GAT' else len(GAT_heads)
        self.dropout = dropout
        self.need_postmp = isinstance(output_dim, tuple)

        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')
        elif self.task == 'graph':
            raise RuntimeError('Graph classification not implemented -- work in progress.')

        # message passing
        self.convs = nn.ModuleList()
        if self.conv_block != 'GAT':  # GAT dimensions are different than others
            self.convs.append(build_conv_layer(self.conv_block, input_dim, hidden_dim))
            if self.need_postmp:
                for _ in range(1, self.num_layers):
                    self.convs.append(build_conv_layer(self.conv_block, hidden_dim, hidden_dim))
            else:
                for _ in range(1, self.num_layers - 1):
                    self.convs.append(build_conv_layer(self.conv_block, hidden_dim, hidden_dim))
                self.convs.append(build_conv_layer(self.conv_block, hidden_dim, output_dim))

        else:
            if self.need_postmp:
                self.convs = build_GAT_conv_layers(input_dim, GAT_hidden_dims, GAT_heads, GAT_concats,
                                                   dropout=dropout)
            else:
                self.convs = build_GAT_conv_layers(input_dim, GAT_hidden_dims + [output_dim], GAT_heads,
                                                   GAT_concats, dropout=dropout)

        # batch normalization
        if self.conv_block == 'GIN':
            self.batch_norms = nn.ModuleList()
            for _ in range(self.num_layers):
                self.batch_norms.append(BatchNorm(hidden_dim))

        # post message passing
        if self.need_postmp:
            if self.conv_block != 'GAT':
                final_hidden_dim = hidden_dim
            else:
                final_hidden_dim = GAT_hidden_dims[-1] * GAT_heads[-1] if GAT_concats[-1] else GAT_hidden_dims[-1]
            self.post_mp = nn.ModuleList()
            for dim in output_dim:
                self.post_mp.append(nn.Linear(final_hidden_dim, dim))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
            raise RuntimeError('No node feature')

        if not self.need_postmp:  # pre-iteration dropout for citation networks (might not be necessary in some case)
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index=edge_index)
            if i != self.num_layers - 1:    # activation and dropout, except for the last iteration
                if self.conv_block == 'GIN':
                    x = self.batch_norms[i](x)
                x = F.relu(x) if self.conv_block != 'GAT' else F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.need_postmp:
            x = F.relu(x) if self.conv_block != 'GAT' else F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            return tuple(self.post_mp[i](x) for i in range(len(self.post_mp)))
        else:
            return x

    def loss(self, pred, label, mask=None):
        if mask is None:
            return F.cross_entropy(pred, label)
        else:
            return sum(F.cross_entropy(pred[i][mask[i], :], label[mask[i]]) for i in range(len(mask))
                       if mask[i].sum().item() > 0)
