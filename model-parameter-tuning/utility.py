# import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, log_loss
import numpy as np
import pandas as pd


# metrics for auc, pr_auc, best_F1, logloss
def cal_metrics_full(gold, pred):
    """calculate metrics given gold and prediction label for binary classification problem
    
    :param gold: true label
    :param pred: predicted label
    :return: 
    roc_auc, pr_auc, best_F1, loss
    """

    roc_auc = roc_auc_score(gold, pred)
    pr_auc = average_precision_score(gold, pred)
    precision, recall, thresholds = precision_recall_curve(gold, pred)
    thresholds = np.append(thresholds, [0])
    pr_curve = pd.DataFrame({"Precision": precision, "Recall": recall, "Thresholds": thresholds})
    pr_curve['F1'] = 2 * pr_curve["Precision"] * pr_curve["Recall"] / (pr_curve["Precision"] + pr_curve["Recall"])
    best_f1 = pr_curve.ix[pr_curve['F1'].argmax()]
    loss = log_loss(y_true=gold, y_pred=pred)
    return roc_auc, pr_auc, best_f1['F1'], loss


# handle floats which should be integers
# works with flat params
def handle_integers(params):
    """mapping integer parameter 
    
    :param params: dict, parameter set 
    :return: processed parameter set
    """
    new_params = {}
    for k, v in params.items():
        if type( v ) == float and int( v ) == v:
            new_params[k] = int( v )
        else:
            new_params[k] = v

    return new_params


# print hidden layers config in readable way
def print_layers(params, logger):
    """print parameter of neural network layer into logger
    
    :param params: parameter space
    :param logger: logger file
    :return: None
    """
    for i in range(1, params['n_layers'] + 1):
        logger.info("layer {} | size: {:>3} | activation: {:<7} | dropout: {:.1%}".
                    format(i,
                           params['layer_{}_size'.format(i)],
                           params['layer_{}_activation'.format(i)],
                           params['layer_{}_dropout'.format(i)]))


def print_params(params, logger):
    """print general parameters into logger without layer parameters
    
    :param params: dict, parameter set
    :param logger: logger file
    :return: 
    """
    logger.info({k: v for k, v in params.items() if not k.startswith('layer_')})
    print_layers(params, logger)

