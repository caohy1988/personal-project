# import function modules from keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Merge, GRU, Masking, Flatten, Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
# import function from utility module
from utility import handle_integers, cal_metrics_full, print_params

def merge_helper(mode, layer_list):
    """meger nueral layers used in residual and highway
    
    :param mode: string, merge type
    :param layer_list: list[layer], can be None list
    :return: 
    merged layers
    """

    layer_list = [item for item in layer_list if item is not None]
    if len(layer_list) >= 2:
        return Merge(mode=mode)(layer_list)
    elif len(layer_list) == 1:
        return layer_list[0]
    else:
        return None


class Validation(Callback):  # inherits from Callback
    """validation is the callback used to early stopping
    
    :param logger: logger
    :param validation_data: tuple of data frame, (valid_data, valid_target)
    :param training_data: tuple of data frame, (training_data, training_target)
    :param metric: metrics used in validating the result, default roc auc
    :param patience: patience used in early stopping, if waiting > patience, then early stopping
    :return: 
    best metric with the round number
    
    """

    def __init__(self, logger, validation_data=(), training_data=(), metric='auc', patience=15):
        # use the callback from Keras
        super(Callback, self).__init__()
        self.metric = metric
        self.patience = patience
        self.X_val, self.y_val = validation_data  # tuple of validation X and y
        self.X_train, self.y_train = training_data # tuple of training X and y
        # initialize the temp result
        self.best = 0.0
        self.best_f1 = 0.0
        self.best_prauc = 0.0
        self.best_loss = 0.0
        self.wait = 0  # counter for patience
        self.best_rounds = 1
        self.counter = 0
        self.logger = logger

    def on_train_begin(self, logs={}):
        """set the auc history
        
        :param logs: dict, input empty log
        :return: 
        history of auc
        """
        self.auc_history = {'valid_auc': [], 'valid_prauc': []}

    def on_epoch_end(self, epoch, logs={}):
        """ fetch the result from epoch
        :param epoch: epoch, number of epoch
        :param logs: dict, log record the loss
        :return: 
        """
        self.counter += 1
        logger = self.logger
        metric = self.metric
        gold_train = self.y_train
        train_data = self.X_train
        # prediction for training data
        pred_train = self.model.predict(train_data, batch_size=32, verbose=0)
        # calculate auc, prauc, best_f1 and loss for training data
        train_auc, train_prauc, train_best_f1, loss_train = cal_metrics_full(gold_train, pred_train)
        gold = self.y_val
        valid_data = self.X_val
        # prediction for validation data
        pred = self.model.predict(valid_data, batch_size=32, verbose=0)
        # calculate auc, prauc, best_f1 and loss for validation data
        valid_auc, valid_prauc, best_f1, loss = cal_metrics_full(gold, pred)
        loss = logs.get("loss")
        # record the metrics
        self.auc_history['valid_auc'].append(valid_auc)
        self.auc_history['valid_prauc'].append(valid_prauc)
        # logger the epoch result
        logger.info(
            "epoch {0:3} loss {1:.4f} t_auc {2:.4f} t_prauc {3:.4f} t_best_f1 {4:.4f} train_loss {5:.4f} "
            "train_auc {6:.4f} train_prauc {7:.4f}, train_best_f1 {8:.4f}".format(
                epoch, logs.get("loss"), valid_auc, valid_prauc, best_f1, loss_train, train_auc, train_prauc,
                train_best_f1))
        if metric == 'auc':
            current = valid_auc
        elif metric == 'prauc':
            current = valid_prauc
        elif metric == 'bestF1':
            current = best_f1
        else:
            current = loss

        logger.info('Metric: {0} | Epoch {1:3} ROC_AUC: {2:.4f} | Best ROC_AUC: {3:.4f} \n'.format(metric, epoch, current, self.best))

        # if improvement over best....
        if metric in ('auc', 'prauc', 'bestF1'):
            if current > self.best:
                self.best = current
                self.best_prauc = valid_prauc
                self.best_loss = loss
                self.best_f1 = best_f1
                self.best_rounds = self.counter
                self.wait = 0
            else:
                if self.wait >= self.patience:  # no more patience, retrieve best model
                    self.model.stop_training = True
                    logger.info(
                        'Best number of rounds: {0:3} \n ROC_AUC: {1:.4f} \n'.format(self.best_rounds, self.best))
                self.wait += 1  # incremental the number of times without improvement
        else:
            # if the metrics is loss
            if current < self.best:
                self.best = current
                self.best_prauc = valid_prauc
                self.best_loss = loss
                self.best_f1 = best_f1
                self.best_rounds = self.counter
                self.wait = 0
            else:
                if self.wait >= self.patience:
                    self.model.stop_training = True
                    logger.info(
                        'Best number of rounds: {0:3} \n ROC_AUC: {1:.4f} \n'.format(self.best_rounds, self.best))
                    self.wait += 1  # incremental the number of times without improvement


def run_func(n_iterations, params, train_data, train_target, valid_data, valid_target, logger):
    """ Building the neural network and running 
    
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
    # set patience for early stopping
    patience = 15
    # set metrics for the key metric evaluation the model
    metric = 'auc'
    # make the params to be integer
    params = handle_integers(params)
    # logger the parameter set
    print_params(params, logger)
    # build the neural network begins
    # define input dimension for the neural network model
    input_dim = train_data.shape[1]
    init = params['init']
    residual = params['residual']
    highway = params['highway']
    num_layer = params['n_layers']
    dnn_output_dim = params['layer_{}_size'.format(num_layer)]
    # initial input layer list
    dnn_inputs = []
    # input initial input dim, specific for future multiple neural network merged model
    dnn_input_total_dim = 0

    # build input layer
    dnn_input = Input(shape=(input_dim,), name='dnn_input')
    dnn_inputs.append(dnn_input)
    dnn_input_total_dim += input_dim
    print('\n ')
    index = 1
    # dummy merge, wait for future rnn or cnn mixed model
    dnn_final_input = merge_helper('concat', dnn_inputs)
    dnn_hid = dnn_final_input
    # initial the highway network skip list
    dnn_skip_connections = []
    # record the input dim
    prev_output_dim = dnn_input_total_dim
    # highway layer construct
    if highway:
        temp_out = dnn_hid
        if prev_output_dim != dnn_output_dim:
            temp_out = Dense(dnn_output_dim)(temp_out)
        # dnn skip list append layer, direct to final layer
        dnn_skip_connections.append(temp_out)
    if dnn_final_input is not None:
        # iterate all layers
        while index < (num_layer + 1):
            layer_input = dnn_hid
            # build through Dense, BatchNormalization, Activation, Dropout
            layer_size = params['layer_{}_size'.format(index)]
            hid_size = layer_size
            layer_activation = params['layer_{}_activation'.format(index)]
            layer_dropout = params['layer_{}_dropout'.format(index)]
            dnn_hid = Dense(layer_size, kernel_initializer=init)(dnn_hid)
            dnn_hid = BatchNormalization()(dnn_hid)
            dnn_hid = Activation(activation=layer_activation)(dnn_hid)
            dnn_hid = Dropout(layer_dropout)(dnn_hid)
            # build residual network
            if residual:
                if prev_output_dim != hid_size:
                    layer_input = Dense(hid_size, kernel_initializer=init)(layer_input)
                dnn_hid = Merge(mode='sum')([dnn_hid, layer_input])
            prev_output_dim = hid_size
            # build highway network
            if highway:
                temp_out = dnn_hid
                if prev_output_dim != dnn_output_dim:
                    temp_out = Dense(dnn_output_dim, init=init)(temp_out)
                dnn_skip_connections.append(temp_out)
            index += 1
        # final merge all highway network
        if highway:
            dnn_hid = merge_helper('sum', dnn_skip_connections)
        else:
            pass
    else:
        dnn_hid = None

    # build output layer
    output = Dense(1)(dnn_hid)
    output = BatchNormalization()(output)
    output = Activation('sigmoid')(output)

    # build model through layer structure
    model = Model(input=dnn_inputs, output=output)
    logger.info("build model completed")
    # compile model
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])
    validation_data = (valid_data, valid_target)
    training_data = (train_data, train_target)
    # build callback
    val_call = Validation(logger=logger,
                          validation_data=validation_data,
                          training_data=training_data,
                          metric=metric,
                          patience=patience)
    # fit model and get training history
    history = model.fit(train_data, train_target,
                        epochs=int(round(n_iterations)),
                        batch_size=params['batch_size'],
                        verbose=0,
                        shuffle=True,
                        validation_data=validation_data,
                        callbacks=[val_call])
    # get the best metrics through model training history
    roc_auc, pr_auc, best_f1, loss = val_call.best, val_call.best_prauc, val_call.best_f1, val_call.best_loss
    # get the best round through training history
    best_round = val_call.best_rounds
    # logger the metrics on log file
    logger.info("# valid  | log loss:{:.4%}, AUC: {:.4%}, PR_AUC: {:.4%}, Best_F1:{:.4%}".format(loss, roc_auc, pr_auc,
                                                                                                 best_f1))
    # log the best round
    logger.info("valid_auc {0:.4f}, best_round {1}".format(roc_auc, best_round))
    return {'loss': roc_auc, 'log_loss': loss, 'auc': roc_auc, 'pr_auc': pr_auc, 'best_f1': best_f1,
            'early_stop': False}

