# Authors: Haiyuan Cao @ Microsoft Research, haca@microsoft.com

# import logging, pprint, numpy, random, time
import numpy as np
import logging
import pprint
import random
from math import log, ceil
from time import time, ctime
# import hyperopt
from hyperopt.pyll.stochastic import sample


def initialize_logger(name, logFilePath, level):
    """Initialize the logger
    
    Given the logger file name, log file path and logging level, create the logger and stream out to console
    :param name: string, log name
    :param logFilePath: string, log file name
    :param level: logging level
    :return: 
    initialized logger
    """
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # specific the logging format
    formatter = logging.Formatter("[%(asctime)s]\t%(levelname)s\t%(message)s", datefmt='%Y-%m-%d %I:%M:%S %p')

    # create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create file handler
    handler = logging.FileHandler(logFilePath)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class Hyperband(object):
    """Using multi-arm bandit algorithm to optimize the hyper-parameter 
    
    The algorithm can obtain optimized configuration of hyperparameters,  
    by given specific metrics, parameter space, running function and training/validation data. 
    The Hyperband algorithm is especially for the machine learning algorithm with the parameter n_epoch. 
    It leverage a sophisticated early stopping strategy according to bandit theory which can save the resource for
    optimization.
    
    Parameters
    ----------
    space: hyperOPT style input dict, necessary parameter space
    run_func: function running for optimization 
    log_file: string, log file name
    max_iter: int, max number of iterations for hyperband algorithm in each trial set 
    eta: int, shrink parameter for each trial set 
    """

    def __init__(self, space, run_func, log_file, max_iter=81, eta=3):
        # define logger
        self.logger = initialize_logger('hyperband', log_file, logging.INFO)
        self.log_file = log_file
        self.space = space
        self.run_func = run_func
        self.max_iter = max_iter  # maximum iterations per configuration
        self.eta = eta  # defines configuration downsampling rate (default = 3)

        # hyperband logic
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))  # number of unique executions of successive halving
        self.B = (self.s_max + 1) * self.max_iter  # total number of iterations (without reuse)

        # initialize counter and result record
        self.results = []
        self.counter = 0
        self.best_loss = -np.inf
        self.best_counter = -1

    def __get_params__(self):
        # sampling the space
        params = sample(self.space)
        return params

    def __try_params__(self, n_iterations, params, train_data, train_target, valid_data, valid_target):
        # using logger to record the running result and space
        logger = self.logger
        logger.info("iterations: " + str(n_iterations))
        run_func = self.run_func
        # try the parameter with input function
        result_dict = run_func(n_iterations, params, train_data, train_target, valid_data, valid_target, logger)
        return result_dict

    def fit(self,  train_data, train_target, valid_data, valid_target, skip_last=0, dry_run=False):
        """Fit api for given trian dataset and valid dataset to get the best parameter set
        
        Given the input training dataset and valid dataset, using hyperband strategy to obtain the best parameter set
        :param train_data: data frame, training data
        :param train_target: data frame, training target
        :param valid_data: data frame, valid data
        :param valid_target: data frame, valid target
        :param skip_last: skip last round or not
        :param dry_run: dry run for test the pipeline using random param set 
        :return: 
        """
        # set the logger
        logger = self.logger

        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            T = [self.__get_params__() for i in range(n)]

            for i in range((s + 1) - int(skip_last)):  # changed from s + 1, if skip last, don't go the last trial set

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                n_iterations = r * self.eta ** i

                # logger number of configurations and numbrer of iterations
                logger.info("\n*** {} configurations x {:.1f} iterations each".format(
                    n_configs, n_iterations))
                # initialize the record of losses
                val_losses = []
                early_stops = []

                # go through n configurations in T
                for t in T:
                    # set counter
                    self.counter += 1
                    # logging best result
                    logger.info("\n{} | {} | best so far: {:.4f} (run {})\n".format(
                        self.counter, ctime(), self.best_loss, self.best_counter))
                    # set start time
                    start_time = time()

                    if dry_run:
                        result = {'loss': random(), 'log_loss': random(), 'auc': random()}
                    else:
                        result = self.__try_params__(n_iterations, t, train_data, train_target, valid_data, valid_target)

                    # count time for running the function
                    seconds = int(round(time() - start_time))
                    # log the seconds
                    logger.info("\n{} seconds.".format(seconds))

                    # record best loss for this run
                    loss = result['loss']
                    val_losses.append(loss)

                    # record early stop or not
                    early_stop = result.get('early_stop', False)
                    early_stops.append(early_stop)

                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey

                    if loss > self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.counter

                    # append counter, time, params and iteration in the record
                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations

                    self.results.append(result)

                # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = np.flipud(np.argsort(val_losses))
                T = [T[i] for i in indices if not early_stops[i]]
                T = T[0:int(n_configs / self.eta)]

        return self

    def get_best(self, n_para):
        """ Given the number of parameters you need, return the top parameter set with the best performance
        
        Given the record of result, sort by the loss and obtain the top parameter set
        :param n_para: number of parameter set need to return 
        :return: 
        Parameter set with the best performance by given specific number
        """
        top_num = n_para
        results = self.results
        logger = self.logger
        log_file_name = self.log_file
        log_file = open(log_file_name, 'a')
        # sort record by loss
        for r in sorted(results, key=lambda x: x['loss'], reverse=True)[:top_num]:
            logger.info("loss: {:.2%} | {} seconds | {:.1f} iterations | run {} ".format(
                r['loss'], r['seconds'], r['iterations'], r['counter']))
            # print the best params into log file
            pprint.pprint(r['params'], log_file)
        return sorted(results, key=lambda x: x['loss'], reverse=True)[:top_num]


