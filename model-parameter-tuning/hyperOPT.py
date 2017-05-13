# Authors: Haiyuan Cao
# Email: haca@microsoft.com

# import package
import logging
import pprint
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials

def initialize_logger(name, logFilePath, level):
    """Initialize the logger

    Given the logger file name, log file path and logging level, create the logger and stream out to console
    :param name: string, log name
    :param logFilePath: string, log file name
    :param level: logging level
    :return: 
    initialized logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
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


def check_epoch_key(params, key):
    """ Check the parameter set contains specific key or not
    
    :param params: dict, parameter set
    :param key: string, name of key
    :return: 
    None
    """
    if key not in params.keys():
        raise KeyError("params should contain " + str(key))


class HyperOPT(object):
    """Hyperopt algorithm: use the Tree Parzen Estimator to optimize the hyper-parameters

    The algorithm can obtain optimized configuration of hyperparameters,  
    by given specific parameter space, running function and training/validation data. 
    The HyperOPT algorithm used here is especially for the machine learning algorithm with the parameter epochew. 
    It use the model based inference to obtain the possible best parameter set
    Parameters
    ----------
    space: hyperOPT style input dict, necessary parameter space
    run_func: function running for optimization 
    log_file: string, log file name

    """

    def __init__(self, space, run_func, log_file):
        # define logger
        self.logger = initialize_logger('hyperopt', log_file, logging.INFO)
        self.log_file = log_file
        self.space = space
        self.run_func = run_func
        # initialize counter and best loss
        self.counter = 0
        self.best_loss = -float('inf')

    def __run_hyperopt__(self, params):
        """Function run in HyperOPT
        Given params through HyperOPT interface, return the result through run function with given parameter set
        
        :param params: 
        :return: dict, contain loss and running status
        """
        # using logger to record the running result and space
        logger = self.logger
        # set counter
        self.counter += 1
        # set the training and valid data and target
        train_data = self.train_data
        train_target = self.train_target
        valid_data = self.valid_data
        valid_target = self.valid_target
        # check the params contains specific key, here check epochs
        check_epoch_key(params=params, key='epochs')
        n_iterations = params['epochs']
        # get result from run function
        result_dict = run_func(n_iterations, params, train_data, train_target, valid_data, valid_target, logger)
        loss = result_dict['loss']
        # record the best loss
        if loss > self.best_loss:
           self.best_loss = loss

        # logger best result with round and best counter in specific run
        logger.info('TPE round {0}, best loss {1:.4f}'.format(self.counter, self.best_loss))

        # return loss
        return {'loss': -1.0 * loss, 'status': STATUS_OK}

    def fit(self, train_data, train_target, valid_data, valid_target, n_iter=100):
        """Fit api for given trian dataset and valid dataset to get the best parameter set

        Given the input training dataset and valid dataset, using hyperband strategy to obtain the best parameter set
        :param train_data: data frame, training data
        :param train_target: data frame, training target
        :param valid_data: data frame, valid data
        :param valid_target: data frame, valid target
        :param n_iter: number of iterations used in HyperOPT
        :return: 
        """
        space = self.space
        # initialize the TPE algorithm
        algo = tpe.suggest
        trials = Trials()
        self.train_data = train_data
        self.train_target = train_target
        self.valid_data = valid_data
        self.valid_target = valid_target
        # running TPE
        self.best_para = fmin(self.__run_hyperopt__, space,
                              algo=algo,
                              max_evals=n_iter,
                              trials=trials)
        # get the best score by sort loss in trial
        self.best_score = max([-t['result']['loss'] for t in trials.trials])
        # get the parameter set list from trials
        self.para_list = trials.trials

    def get_best(self, n_para):
        """ Given the number of parameters you need, return the top parameter set with the best performance

        Given the record of result, sort by the loss and obtain the top parameter set
        :param n_para: number of parameter set need to return 
        :return: 
        Parameter set with the best performance by given specific number
        """
        logger = self.logger
        top_num = n_para
        para_list = self.para_list
        best_score = self.best_score
        best_para = self.best_para
        log_file_name = self.log_file
        # sort the result dict by loss
        sort_list = sorted(para_list, key=lambda t: t['result']['loss'])[:top_num]
        logger.info('best score: {0}'.format(best_score))
        logger.info('best para')
        log_file = open(log_file_name, 'a')
        pprint.pprint(best_para, log_file)
        # print the best parameter set and score to file
        for para_dict in sort_list[1:]:
            pprint.pprint(para_dict['result'], log_file)
            pprint.pprint(para_dict['misc']['vals'], log_file) 

