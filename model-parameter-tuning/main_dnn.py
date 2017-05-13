# Author: Haiyuan Cao @ Microsoft Research, haca@microsoft.com
# import package pandas, numpy, hyperopt
import pandas as pd
import numpy as np
from hyperopt import hp

# import the customized function for optimization
from run_func import run_func

# import optimization strategy
from hyperOPT import HyperOPT
from hyperband import Hyperband


def main():
    """Input for the parameter optimization process
    
    Given the parameter space, running function and training/valid dataset, 
    using hyperband/hyperopt strategy to find the top parameter set with the best metrics 
    :input:
    ------                                              
    run_func: running function used in optimization 
    log_file_name: string, name of log file
    top_num: int, number of best configurations you want to choose
    train_train: data frame, the last column should be the target value, training data set in optimization process
    train_valid: data frame, the last column should be the target value, validation data set in optimization process
    :return: 
    --------
    top parameters and their score on log file
    
    """
    # define the log file name for hyperopt method
    log_file_hyperopt = 'hyperopt.txt'
    # set the number of parameter sets to be recorded
    top_num = 2
    # set the input file name
    train_train = pd.read_csv('fargo_train_train.csv')
    train_valid = pd.read_csv('fargo_train_valid.csv')

    # shuffle the input file
    train_train_shuffle = train_train.reindex(np.random.permutation(train_train.index)).sort_index()
    train_valid_shuffle = train_valid.reindex(np.random.permutation(train_valid.index)).sort_index()

    # split the dataset into data and target
    train_data, train_target = train_train_shuffle.values[:, 0:-1].astype(np.float32), train_train_shuffle.values[:, -1]
    valid_data, valid_target = train_valid_shuffle.values[:, 0:-1].astype(np.float32), train_valid_shuffle.values[:, -1]

    # define parameter space, must contain epochs !
    max_layers = 6

    space_hyperopt = {
        'n_layers': hp.quniform('n_layers', 2, max_layers, 1),
        'init': hp.choice('init', ('uniform', 'normal', 'glorot_uniform',
                                'glorot_normal')),
        'batch_size': hp.choice('batch_size', (16, 32, 64, 128)),
        'epochs': hp.choice('epochs', [1, 2]),
        'optimizer': 'adam',
        'residual': hp.choice('residual', [True, False]),
        'highway': hp.choice('highway', [True, False]),
    }

    # for each hidden layer, we choose size, activation and extras individually
    for i in range(1, max_layers + 1):
        space_hyperopt['layer_{}_size'.format(i)] = hp.quniform('ls{}'.format(i), 20, 200, 20)
        space_hyperopt['layer_{}_activation'.format(i)] = 'relu'
        space_hyperopt['layer_{}_dropout'.format(i)] = hp.quniform('dropout{}'.format(i), 0.0, 0.5, 0.05)

    # using HyperOPT to optimize the parameter set
    para_hyperopt = HyperOPT(space=space_hyperopt, run_func=run_func, log_file=log_file_hyperopt)

    # using fit api to fit the parameter set
    para_hyperopt.fit(train_data=train_data,
                      train_target=train_target,
                      valid_data=valid_data,
                      valid_target=valid_target,
                      n_iter=2)
    # using get best to take best parameter set
    para_hyperopt.get_best(top_num)
    
    # set the log file name for hyperband method
    log_file_hyperband = 'hyperband.txt'

    # define parameter set, must contain epochs !
    space_hyperband = {
        'n_layers': hp.quniform('n_layers', 2, max_layers, 1),
        'init': hp.choice('init', ('uniform', 'normal', 'glorot_uniform',
                                   'glorot_normal')),
        'batch_size': hp.choice('batch_size', (16, 32, 64, 128)),
        'optimizer': 'adam',
        'residual': hp.choice('residual', [True, False]),
        'highway': hp.choice('highway', [True, False]),
    }

    # for each hidden layer, we choose size, activation and extras individually
    for i in range(1, max_layers + 1):
        space_hyperband['layer_{}_size'.format(i)] = hp.quniform('ls{}'.format(i), 20, 200, 20)
        space_hyperband['layer_{}_activation'.format(i)] = 'relu'
        space_hyperband['layer_{}_dropout'.format(i)] = hp.quniform('dropout{}'.format(i), 0.0, 0.5, 0.05)

    # using Hyperband to optimize the parameter set
    para_hyperband = Hyperband(space=space_hyperband, run_func=run_func, log_file=log_file_hyperband, max_iter=4, eta=2)

    # using fit api to fit the parameter set
    para_hyperband.fit(train_data=train_data,
                       train_target=train_target,
                       valid_data=valid_data,
                       valid_target=valid_target)
    # using get best to take best parameter set
    para_hyperband.get_best(top_num)
    

if __name__ == "__main__":
    main()
