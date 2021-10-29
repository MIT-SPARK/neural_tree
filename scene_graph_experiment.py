"""
This script computes results for the scene graph experiments using the Neural Tree (message passing on H-trees) or the
vanilla architectures (message passing on original graphs).
The dataset split is generated randomly for each run with fixed random seed.
"""
from neural_tree.utils.base_training_job import BaseTrainingJob, print_log
from neural_tree.dataset_loader import StanfordDataset
from statistics import mean, stdev
from os import mkdir, path
import random
from datetime import datetime

############## dataset ############################################
project_dir = path.dirname(path.abspath(__file__))
dataset = StanfordDataset(project_dir + '/data/Stanford3DSG.pkl')
task = 'node'

############## run control #########################################
num_runs = 100
early_stop_window = -1  # setting to -1 will disable early stop
verbose = False
log_folder_master = project_dir + '/log'

############### algorithm ##########################################
algorithm = 'neural_tree'
# algorithm = 'original'

############### parameters #########################################
train_node_ratio = 0.7
val_node_ratio = 0.1
test_node_ratio = 0.2
network_params = {'conv_block': 'GraphSAGE',
                  'hidden_dim': 128,
                  'num_layers': 4,
                  'GAT_hidden_dims': [128, 128],
                  'GAT_heads': [6, 1],
                  'GAT_concats': [True, False],
                  'dropout': 0.25}
optimization_params = {'lr': 0.005,
                       'num_epochs': 1000,
                       'weight_decay': 0.001}
dataset_params = {'batch_size': 128,
                  'shuffle': True}
neural_tree_params = {'min_diameter': 1,      # diameter=0 means the H-tree is disconnected
                      'max_diameter': None,
                      'sub_graph_radius': None}

#####################################################################

if __name__ == '__main__':
    random.seed(0)

    # setup log folder, parameter and accuracy files
    log_folder = log_folder_master + datetime.now().strftime('/%Y%m%d-%H%M%S')
    mkdir(log_folder)
    print('Starting node classification on Stanford 3D Scene Graph dataset using {}. Results saved to {}'.
          format(algorithm, log_folder))
    f_param = open(log_folder + '/parameter.txt', 'w')
    f_log = open(log_folder + '/accuracy.txt', 'w')

    print('train_node_ratio: {}'.format(train_node_ratio), file=f_param)
    print('val_node_ratio: {}'.format(val_node_ratio), file=f_param)
    print('test_node_ratio: {}'.format(test_node_ratio), file=f_param)

    # run experiment
    test_accuracy_list = []
    val_accuracy_list = []
    for i in range(num_runs):
        print("run number: ", i)

        # data split
        dataset.generate_node_split(train_node_ratio=train_node_ratio, val_node_ratio=val_node_ratio,
                                    test_node_ratio=test_node_ratio)

        # training
        train_job = BaseTrainingJob(algorithm, task, dataset, network_params, neural_tree_params)
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
    max_val_accuracy = max(val_accuracy_list)
    print_log('Best validation accuracy over all runs: {:.2f} %, corresponding test accuracy: {:.2f} %'.
              format(max_val_accuracy, test_accuracy_list[val_accuracy_list.index(max_val_accuracy)]), file=f_log)
    f_log.close()
    print('End of scene_graph_experiment.py. Results saved to: {}\n'.format(log_folder))
