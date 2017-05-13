# import sklearn, numpy, pandas, xgboost
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, log_loss
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier

# logging parameters in xgboost model


def print_params(params, logger):
    """ print parameters of model 
    
    :param params: parameter set of model
    :param logger: logger used in record the training result
    :return: 
    logger with parameter of model
    """
    logger.info({k: v for k, v in params.items() if not k.startswith('layer_')})

# metrics for auc, pr_auc, best_F1, logloss


def cal_metrics_full(gold, pred):
    """calculate the metrics of model including, auc, prauc, log_loss and best F1
    
    :param gold: truth label
    :param pred: preidiction label
    :return: 
    roc_auc, pr_auc, best_f1, loss
    """
    # roc auc score
    roc_auc = roc_auc_score(gold, pred)
    # pr auc score
    pr_auc = average_precision_score(gold, pred)
    precision, recall, thresholds = precision_recall_curve(gold, pred)
    thresholds = np.append(thresholds, [0])
    pr_curve = pd.DataFrame({"Precision": precision, "Recall": recall, "Thresholds": thresholds})
    pr_curve['F1'] = 2 * pr_curve["Precision"] * pr_curve["Recall"] / (pr_curve["Precision"] + pr_curve["Recall"])
    # best F1
    best_f1 = pr_curve.ix[pr_curve['F1'].argmax()]
    # log loss
    loss = log_loss(y_true=gold, y_pred=pred)
    return roc_auc, pr_auc, best_f1['F1'], loss


# handle floats which should be integers
# works with flat params
def handle_integers( params ):
    """handle the integers in parameters of model
    
    :param params: parameter set of model
    :return: 
    parameter set with integer parameter
    """

    new_params = {}
    for k, v in params.items():
        if type(v) == float and int(v) == v:
            new_params[k] = int(v)
        else:
            new_params[k] = v
    return new_params


def run_func(n_iterations, params, train_data, train_target, valid_data, valid_target, logger):
    """ Building boost tree model and running
    
    :param n_iterations: epoch number used in neural network iterations
    :param params: parameter set used for build neural network
    :param train_data: training dataset for parameter optimization
    :param train_target: training target for parameter optimization 
    :param valid_data: valid dataset for parameter optimization 
    :param valid_target: valid target for parameter optimization 
    :param logger: logger used to record the temporary from function running
    :return: 
    dict of all metrics, must contain the key loss
    """
    # set metric
    metric = 'auc'
    params = handle_integers(params)
    # set n_estimator in parameter
    params['n_estimators'] = n_iterations
    # delete epoch key
    params.pop('epochs', None)
    # logger parameter in logger
    print_params(params, logger)
    # training xgboost model
    model = XGBClassifier(**params)
    # set validation set
    eval_set = [(valid_data, valid_target)]
    # set early stopping round, defaul 10% of total round
    early_stopping_rounds = int(n_iterations * 0.1)
    # fit model
    model.fit(train_data, train_target,
              early_stopping_rounds=early_stopping_rounds,
              eval_metric=metric,
              eval_set=eval_set,
              verbose=True)
    # predict result
    pred_target = model.predict_proba(valid_data)[:, 1]
    # get metrics from prediction result
    roc_auc, pr_auc, best_f1, loss = cal_metrics_full(valid_target, pred_target)
    # logger the result
    logger.info("# valid  | log loss:{:.4%}, AUC: {:.4%}, PR_AUC: {:.4%}, Best_F1:{:.4%}".format(loss, roc_auc, pr_auc,
                                                                                                 best_f1))
    # logger the best valid metrics
    logger.info("best valid_auc {0:.4f}".format(roc_auc))
    return {'loss': roc_auc, 'log_loss': loss, 'auc': roc_auc, 'pr_auc': pr_auc, 'best_f1': best_f1,
            'early_stop': False}

