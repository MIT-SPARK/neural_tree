import torch
import random


def generate_mask(dataset, train_graph_ratio, val_graph_ratio):
    assert train_graph_ratio > 0
    assert val_graph_ratio >= 0
    assert train_graph_ratio + val_graph_ratio <= 1

    num_graphs = len(dataset)
    random_idx_list = list(range(num_graphs))
    random.shuffle(random_idx_list)

    end_train_idx = int(len(dataset) * train_graph_ratio)
    end_val_idx = end_train_idx + int(len(dataset) * val_graph_ratio)

    # train graphs
    for i in random_idx_list[: end_train_idx]:
        num_nodes = dataset[i].x.shape[1]
        dataset[i].train_mask = torch.ones(num_nodes, dtype=torch.bool)
        dataset[i].val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        dataset[i].test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # validation graphs
    for i in random_idx_list[end_train_idx: end_val_idx]:
        num_nodes = dataset[i].x.shape[1]
        dataset[i].train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        dataset[i].val_mask = torch.ones(num_nodes, dtype=torch.bool)
        dataset[i].test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # test graphs
    for i in random_idx_list[end_val_idx:]:
        num_nodes = dataset[i].x.shape[1]
        dataset[i].train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        dataset[i].val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        dataset[i].test_mask = torch.ones(num_nodes, dtype=torch.bool)
    return dataset


def generate_node_split(data, num_classes, num_train_per_class, num_val, num_test):
    data['train_mask'] = torch.zeros(data.num_nodes, dtype=torch.bool)
    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
        data.train_mask[idx] = True

    remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.size(0))]

    data['val_mask'] = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[remaining[:num_val]] = True

    data['test_mask'] = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[remaining[num_val:num_val + num_test]] = True
    return data
