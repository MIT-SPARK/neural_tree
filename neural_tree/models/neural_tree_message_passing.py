from neural_tree.models import BasicNetwork
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class NeuralTreeNetwork(BasicNetwork):
    def __init__(self, input_dim, output_dim, task='node', conv_block='GCN', hidden_dim=None, num_layers=None,
                 GAT_hidden_dims=None, GAT_heads=None, GAT_concats=None, dropout=0.25):
        """
        NeuralTreeNetwork is the child class of BasicNetwork, which implements basic message passing on graphs.
        The network parameters and loss functions are the same as the parent class. The difference is that this class
         has an additional pooling layer at the end to aggregate final hidden states of the leaf nodes (for node
         classification).
        """
        super(NeuralTreeNetwork, self).__init__(input_dim, output_dim, task, conv_block, hidden_dim, num_layers,
                                                GAT_hidden_dims, GAT_heads, GAT_concats, dropout)

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

        x = pyg_nn.global_mean_pool(x[data.leaf_mask, :], batch[data.leaf_mask])

        if self.need_postmp:
            x = F.relu(x) if self.conv_block != 'GAT' else F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            return tuple(self.post_mp[i](x) for i in range(len(self.post_mp)))
        else:
            return x
