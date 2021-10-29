"""
This script computes results for the citation experiments using the Neural Tree (message passing on H-trees) or the
vanilla architectures (message passing on original graphs).
For the Neural Tree, you need to pre-process the raw data to generate H-trees by running
preprocess_dataset_with_subsampling.py with desired tree width. The original dataset and the H-trees are saved in
    preprocessed_<dataset_name>/<dataset_name>_tw<tree_width>.pkl.

The data split will be randomly generated and saved to
    preprocessed_<dataset_name>/train<num_train_per_class>_<run_idx>.pkl.
If this file already exist, this script get data split from the existing file directly. Note that the data split is
independent of treewidth.
"""
from neural_tree.utils.base_training_job import BaseTrainingJob, print_log
from neural_tree.h_tree import get_subtrees_from_htree, HTreeDataset
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from statistics import mean, stdev
from os import mkdir, path
from datetime import datetime
import time
import random
import pickle

############## dataset #############################################
experiment_dir = path.dirname(path.abspath(__file__))
dataset_name = 'pubmed'
preprocessed_tree_dir = experiment_dir + '/preprocessed_{}'.format(dataset_name)

############## run control #########################################
num_runs = 10
early_stop_window = -1  # setting to -1 will disable early stop
verbose = True
log_folder_master = experiment_dir + datetime.now().strftime('/log/%Y%m%d-%H%M%S')

############### algorithm ##########################################
treewidth = 1
algorithm = 'neural_tree'
# algorithm = 'original'

############### parameters #########################################
num_train_per_class_list = [20, 400, 800, 1200, 1600, 2000]
network_params = {'conv_block': 'GCN',
                  'hidden_dim': 16,
                  'num_layers': 2,
                  'GAT_hidden_dims': [8],
                  'GAT_heads': [8, 8],
                  'GAT_concats': [True, False],
                  'dropout': 0.5}
optimization_params = {'lr': 0.005,
                       'num_epochs': 300,
                       'weight_decay': 0.0005}
dataset_params = {'batch_size': 128,
                  'shuffle': True}
neural_tree_params = {'min_diameter': 0,    # sub-tree might be disconnected if num_layers is less than treewidth
                      'max_diameter': None,
                      'sub_graph_radius': None}

#####################################################################

if __name__ == '__main__':
    random.seed(0)
    mkdir(log_folder_master)

    # preprocessed H-tree
    if algorithm == 'neural_tree':
        tree_decomposition_file_path = preprocessed_tree_dir + '/{}_tw{}.pkl'.format(dataset_name, treewidth)
        if not path.exists(tree_decomposition_file_path):
            raise RuntimeError('Cannot find {}. Make sure you have run preprocess_citation_dataset_with_subsampling.py '
                               'to generate the preprocessed H-tree decomposition and set the dataset directory, '
                               'preprocessed_tree_dir, correctly.'.format(tree_decomposition_file_path))
        with open(tree_decomposition_file_path, 'rb') as input_file:
            saved_data = pickle.load(input_file)
            G_jth_list = saved_data['tree_decomposition']
        print('Loaded tree decomposition from:', tree_decomposition_file_path)

    # run experiment
    for num_train_per_class in num_train_per_class_list:
        log_folder = log_folder_master + \
                     '/{}{}_{}'.format(dataset_name, num_train_per_class, network_params['conv_block']) + \
                     ('_NT_tw{}'.format(treewidth) if algorithm == 'neural_tree' else '')

        print('Starting node classification on {} dataset using {}. Results saved to {}'.
              format(dataset_name, algorithm, log_folder))
        mkdir(log_folder)
        f_param = open(log_folder + '/parameter.txt', 'w')
        f_log = open(log_folder + '/accuracy.txt', 'w')

        test_accuracy_list = []
        val_accuracy_list = []
        for i in range(num_runs):
            print("run number: ", i)

            # data split
            dataset_split_file_path = preprocessed_tree_dir + '/train{}_{}.pkl'.format(num_train_per_class, i)
            if path.exists(dataset_split_file_path):
                with open(dataset_split_file_path, 'rb') as input_file:
                    dataset = pickle.load(input_file)
                    assert dataset[0].train_mask.sum().item() == num_train_per_class * dataset.num_classes
                print('Load tree dataset split from:', dataset_split_file_path)
            else:
                dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name, split="random",
                                    num_train_per_class=num_train_per_class, num_val=500, num_test=1000,
                                    transform=T.NormalizeFeatures())
                with open(dataset_split_file_path, 'wb') as output_file:
                    pickle.dump(dataset, output_file, pickle.HIGHEST_PROTOCOL)
                print('Save tree dataset split to:', dataset_split_file_path)

            if algorithm == 'neural_tree':
                tic = time.perf_counter()
                train_list_all = []
                val_list_all = []
                test_list_all = []
                num_convs = len(network_params['GAT_heads']) if network_params['conv_block'] == 'GAT' \
                    else network_params['num_layers']
                for G_jth in G_jth_list:
                    train_list, val_list, test_list = get_subtrees_from_htree(dataset[0], G_jth, num_convs)
                    train_list_all += train_list
                    val_list_all += val_list
                    test_list_all += test_list
                dataset = HTreeDataset([train_list_all, val_list_all, test_list_all], dataset.num_node_features,
                                       dataset.num_classes, dataset.name + '_h-tree', 'node')
                toc = time.perf_counter()
                print('Done generating random dataset split (time elapsed {:.1f} s).'.format(toc - tic))

            train_job = BaseTrainingJob(algorithm, 'node', dataset, network_params, neural_tree_params)
            model, best_acc = train_job.train(log_folder + '/' + str(i), optimization_params, dataset_params,
                                              early_stop_window=early_stop_window, verbose=verbose)
            if i == 0:
                # save parameters in train_job to parameter.txt (only need to save once)
                train_job.print_training_params(f=f_param)
                f_param.close()

            if isinstance(best_acc, tuple):
                val_accuracy_list.append(best_acc[0] * 100)
                test_accuracy_list.append(best_acc[1] * 100)
            else:
                raise RuntimeError('Validation set is empty.')

        print_log('Validation accuracy: {}'.format(val_accuracy_list), f_log)
        print_log('Test accuracy: {}'.format(test_accuracy_list), f_log)
        print_log("Average test accuracy from best validation accuracy ({:.2f} +/- {:.2f} %) over {} runs is "
                  "{:.2f} +/- {:.2f} %. ".format(mean(val_accuracy_list), stdev(val_accuracy_list), num_runs,
                                                 mean(test_accuracy_list), stdev(test_accuracy_list)), f_log)
        f_log.close()
    print('End of citation_increasing_data.py. Results saved to:', log_folder_master)

