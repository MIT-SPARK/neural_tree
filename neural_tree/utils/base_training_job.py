from neural_tree.models import BasicNetwork, NeuralTreeNetwork
from neural_tree.h_tree import convert_dataset_to_junction_tree_hierarchy, HTreeDataset
import sys
import time
from copy import deepcopy
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tensorboardX import SummaryWriter


class BaseTrainingJob:
    def __init__(self, network_name, task, dataset, network_params=None, neural_tree_params=None):
        assert task == 'node' or task == 'graph'
        assert network_name == 'original' or 'neural_tree'
        self.__network_name = network_name
        self.__task = task

        # initialize training parameters
        if dataset.num_node_features == 0:
            raise RuntimeError('No node feature')
        self.__training_params = self.create_default_params(network_name)
        self.update_training_params(network_params=network_params)
        self.update_training_params(network_params={'input_dim': dataset.num_node_features,
                                                    'output_dim': dataset.num_classes})
        if self.__network_name == 'neural_tree':
            self.update_training_params(neural_tree_params=neural_tree_params)
        self.remove_unused_network_params(self.__training_params['network_params'])

        # save self.__dataset
        if self.__network_name == 'original':
            assert isinstance(dataset, HTreeDataset) is False
            self.__dataset = dataset

        # convert dataset for junction tree hierarchy
        else:  # self.__network_name == 'neural_tree':
            min_diameter = self.__training_params['neural_tree_params']['min_diameter']
            max_diameter = self.__training_params['neural_tree_params']['max_diameter']
            radius = self.__training_params['neural_tree_params']['sub_graph_radius']

            if self.__task == 'node':
                if isinstance(dataset, HTreeDataset):  # use saved dataset file to save time
                    print('Parsed {} graphs for training, {} for validation, {} for testing from the original dataset.' \
                          .format(len(dataset.dataset_jth[0]), len(dataset.dataset_jth[1]),
                                  len(dataset.dataset_jth[2])))
                    dataset_filtered = []
                    for data_list in dataset.dataset_jth:  # HTreeDataset has three Data lists in dataset_jth
                        dataset_filtered.append([data for data in data_list if
                                                 (min_diameter is None or data.diameter >= min_diameter) and
                                                 (max_diameter is None or data.diameter <= max_diameter)])
                        self.__dataset = dataset_filtered
                else:
                    print('Received {} graphs for node classification.'.format(len(dataset)))
                    tic = time.perf_counter()
                    self.__dataset = convert_dataset_to_junction_tree_hierarchy(dataset, self.__task,
                                                                                min_diameter=min_diameter,
                                                                                max_diameter=max_diameter,
                                                                                radius=radius)
                    toc = time.perf_counter()
                    print('Done computing junction tree hierarchies (time elapsed: {:.4f} s). '.format(toc - tic))
                print('After diameter filtering, got {} graphs for training, {} for validation, {} for testing.' \
                      .format(len(self.__dataset[0]), len(self.__dataset[1]), len(self.__dataset[2])))
                max_diameter_in_dataset = max(map(lambda data_list: max([data.diameter for data in data_list]),
                                                  self.__dataset))
                print('Maximum junction tree diameter in the dataset: {}.'.format(max_diameter_in_dataset))
            else:  # self.__task == 'graph'
                raise RuntimeError('Graph classification not implemented.')

        # initialize network
        self.__net = self.initialize_network()

    @staticmethod
    def create_default_params(network_name):
        network_params = {'input_dim': None,
                          'output_dim': None,
                          'hidden_dim': 32,
                          'num_layers': 3,
                          'dropout': 0.25,
                          'conv_block': 'GCN',
                          'GAT_hidden_dims': None,
                          'GAT_heads': None,
                          'GAT_concats': None}
        optimization_params = {'lr': 0.01,
                               'num_epochs': 200}
        dataset_params = {'batch_size': 64,
                          'shuffle': True}
        neural_tree_params = {'min_diameter': 1,
                              'max_diameter': None,
                              'sub_graph_radius': 2}
        if network_name == 'original':
            training_params = {'network_params': network_params, 'optimization_params': optimization_params,
                               'dataset_params': dataset_params}
        else:
            training_params = {'network_params': network_params, 'optimization_params': optimization_params,
                               'dataset_params': dataset_params, 'neural_tree_params': neural_tree_params}
        return training_params

    @staticmethod
    def remove_unused_network_params(network_params):
        if network_params['conv_block'] != 'GAT':
            removed_params = ['GAT_hidden_dims', 'GAT_heads', 'GAT_concats']
        else:
            removed_params = ['hidden_dim', 'num_layers']
        for param in removed_params:
            network_params.pop(param)
        return network_params

    def print_training_params(self, f=sys.stdout):
        for params, params_dict in self.__training_params.items():
            print(params, file=f)
            for param_name, value in params_dict.items():
                print('   {}: {}'.format(param_name, value), file=f)

    def update_training_params(self, network_params=None, optimization_params=None, dataset_params=None,
                               neural_tree_params=None):
        if network_params is not None:
            for key in network_params:
                self.__training_params['network_params'][key] = network_params[key]
        if optimization_params is not None:
            for key in optimization_params:
                self.__training_params['optimization_params'][key] = optimization_params[key]
        if dataset_params is not None:
            for key in dataset_params:
                self.__training_params['dataset_params'][key] = dataset_params[key]
        if neural_tree_params is not None:
            for key in neural_tree_params:
                self.__training_params['neural_tree_params'][key] = neural_tree_params[key]

    def initialize_network(self):
        if self.__network_name == 'original':
            return BasicNetwork(task=self.__task, **self.__training_params['network_params'])
        if self.__network_name == 'neural_tree':
            return NeuralTreeNetwork(task=self.__task, **self.__training_params['network_params'])
        else:
            raise RuntimeError('Unknown network name.')

    def get_dataset(self):
        return self.__dataset

    def train(self, log_folder, optimization_params=None, dataset_params=None, decay_epochs=100, decay_rate=1.0,
              early_stop_window=-1, verbose=False):
        # update parameters
        self.update_training_params(optimization_params=optimization_params, dataset_params=dataset_params)

        # move training to gpu if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__net.to(device)
        # create data loader
        if self.__task == 'node':
            if self.__network_name == 'neural_tree':
                train_loader = DataLoader(self.__dataset[0],
                                          batch_size=self.__training_params['dataset_params']['batch_size'],
                                          shuffle=self.__training_params['dataset_params']['shuffle'])
                val_loader = DataLoader(self.__dataset[1],
                                        batch_size=self.__training_params['dataset_params']['batch_size'],
                                        shuffle=self.__training_params['dataset_params']['shuffle'])
                test_loader = DataLoader(self.__dataset[2],
                                         batch_size=self.__training_params['dataset_params']['batch_size'],
                                         shuffle=self.__training_params['dataset_params']['shuffle'])
            else:
                train_loader = val_loader = test_loader = \
                    DataLoader(self.__dataset, batch_size=self.__training_params['dataset_params']['batch_size'],
                               shuffle=self.__training_params['dataset_params']['shuffle'])
        else:  # self.__task == 'graph'
            raise RuntimeError('Graph classification not implemented.')

        opt = optim.Adam(self.__net.parameters(), lr=self.__training_params['optimization_params']['lr'],
                         weight_decay=self.__training_params['optimization_params']['weight_decay'])

        my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=decay_epochs, gamma=decay_rate)

        # train
        max_val_acc = 0
        max_test_acc = 0  # If val_loader is not None, compute using the model weights that lead to best_val_acc
        best_model_state = None
        writer = SummaryWriter(log_folder)
        if val_loader is None:
            early_stop_window = -1  # do not use early stopping if there's no validation set

        tic = time.perf_counter()
        early_stop_step = 0
        for epoch in range(self.__training_params['optimization_params']['num_epochs']):
            early_stop_step += 1
            total_loss = 0.
            self.__net.train()
            for batch in train_loader:
                opt.zero_grad()

                pred = self.__net(batch.to(device))
                label = batch.y
                if self.__task == 'node' and self.__network_name == 'original':
                    if isinstance(self.__training_params['network_params']['output_dim'], tuple):
                        pred = tuple(pred_i[batch.train_mask] for pred_i in pred)
                        loss = self.__net.loss(pred, label[batch.train_mask], (batch.y_room[batch.train_mask],
                                                                               batch.y_object[batch.train_mask]))
                    else:   # citation dataset
                        loss = self.__net.loss(pred[batch.train_mask], label[batch.train_mask])
                elif isinstance(self.__training_params['network_params']['output_dim'], tuple):
                    loss = self.__net.loss(pred, label, (batch.y_room, batch.y_object))
                else:
                    loss = self.__net.loss(pred, label)

                loss.backward()
                opt.step()

                total_loss += loss.item() * batch.num_graphs
            total_loss /= len(train_loader.dataset)
            writer.add_scalar('loss', total_loss, epoch)

            writer.add_scalar('lr', opt.param_groups[0]["lr"], epoch)
            my_lr_scheduler.step()

            if verbose:
                train_result = self.test(train_loader, is_train=True)
                writer.add_scalar('train result', train_result, epoch)

            # validation and testing
            if val_loader is not None:
                val_result = self.test(val_loader, is_validation=True)
                writer.add_scalar('validation result', val_result, epoch)
                if val_result > max_val_acc:
                    max_val_acc = val_result
                    best_model_state = deepcopy(self.__net.state_dict())
                    early_stop_step = 0
                if verbose and (epoch + 1) % 10 == 0:
                    print('Epoch {:03}. Loss: {:.4f}. Train accuracy: {:.4f}. Validation accuracy: {:.4f}.'
                          .format(epoch, total_loss, train_result, val_result))
                if early_stop_step == early_stop_window and epoch > early_stop_window:
                    if verbose:
                        print('Early stopping condition reached at {} epoch.'.format(epoch))
                    break
            else:
                test_result = self.test(test_loader)
                writer.add_scalar('test result', test_result, epoch)
                if test_result > max_test_acc:
                    max_test_acc = test_result
                    best_model_state = deepcopy(self.__net.state_dict())
                if verbose and (epoch + 1) % 10 == 0:
                    print('Epoch {:03}. Loss: {:.4f}. Train accuracy: {:.4f}. Test accuracy: {:.4f}.'.
                          format(epoch, total_loss, train_result, test_result))

        toc = time.perf_counter()
        print('Training completed (time elapsed: {:.4f} s). '.format(toc - tic))

        self.__net.load_state_dict(best_model_state)

        if val_loader is not None:
            tic = time.perf_counter()
            test_result = self.test(test_loader)
            toc = time.perf_counter()
            print('Testing completed (time elapsed: {:.4f} s). '.format(toc - tic))
            print('Best validation accuracy: {:.4f}, corresponding test accuracy: {:.4f}.'.
                  format(max_val_acc, test_result))
            return self.__net, (max_val_acc, test_result)
        else:
            print('Best test accuracy: {:.4f}.'.format(max_test_acc))
            return self.__net, max_test_acc

    def test(self, data_loader, is_train=False, is_validation=False):
        self.__net.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        correct = 0
        for data in data_loader:
            with torch.no_grad():
                pred = self.__net(data.to(device))
                if isinstance(self.__training_params['network_params']['output_dim'], tuple):  # assume two node types
                    pred_room = pred[0].argmax(dim=1)
                    pred = pred[1].argmax(dim=1)  # object
                    pred[data.y_room] = pred_room[data.y_room]
                else:
                    pred = pred.argmax(dim=1)
                label = data.y

            if self.__task == 'node' and self.__network_name == 'original':
                if is_train:
                    mask = data.train_mask
                elif is_validation:
                    mask = data.val_mask
                else:
                    mask = data.test_mask
                pred = pred[mask]
                label = data.y[mask]

            correct += pred.eq(label).sum().item()

        if self.__task == 'node' and self.__network_name == 'original':
            if is_train:
                total = sum([torch.sum(data.train_mask).item() for data in data_loader.dataset])
            elif is_validation:
                total = sum([torch.sum(data.val_mask).item() for data in data_loader.dataset])
            else:
                total = sum([torch.sum(data.test_mask).item() for data in data_loader.dataset])
        else:
            total = len(data_loader.dataset)

        return correct / total


def print_log(string, file):
    print(string)
    print(string, file=file)
