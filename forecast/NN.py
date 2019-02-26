#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from keras.layers import Dense, Activation, BatchNormalization, Dropout, LSTM, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras import backend as K
from keras import optimizers

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import class_weight

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

class Model:
    def __init__(self, X_train, y_train, X_val, y_val):
       #K.clear_session()    
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.metrics = Metrics()
        self.model = Sequential()
        self.class_weights = dict(enumerate(
            class_weight.compute_class_weight('balanced',
                                            np.unique(y_train),
                                            y_train)))

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def f1_loss(self, y_true, y_pred):
        
        tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
        tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2*p*r / (p+r+K.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        return 1 - K.mean(f1)

    def nn_fit(self):
        print('Feeforward Neural Network Training Started')
        save_fname = os.path.join(
            'saved_models', 
            '%s-%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'),str('nn'))
        )

        callbacks = [
            self.metrics,
			EarlyStopping(monitor='val_loss', patience=10),
			ModelCheckpoint(filepath=save_fname, 
            monitor='val_loss', save_best_only=True), 
            TensorBoard
		]
        self.model.add(Dense(100, activation= 'relu', 
        input_shape=(self.X_train.shape[1], )))
        self.model.add(Dense(10, activation= 'relu'))
        self.model.add(Dense(20, activation= 'relu'))
        self.model.add(Dropout(0.74))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary()
        self.model.compile(optimizer='adam',
            loss=  'binary_crossentropy', 
            metrics=['accuracy']
            )
            
        history = self.model.fit(self.X_train,
                    self.y_train,
                    epochs = 1,
                    batch_size=16,
                    class_weight = self.class_weights, 
                    validation_data = (self.X_val, self.y_val), 
                    callbacks = callbacks, 
                    verbose=1)

        self.model.save(save_fname)

        return history

    def lstm_fit(self):
        NAME = 'LSTM' 
        print(f'{NAME} Neural Network Training Started')
        save_fname = os.path.join(
            'saved_models', 
            '%s-%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'),str('lstm'))
            )

        tensorboard = TensorBoard(log_dir=f'logs-{NAME}')
        callbacks = [
            self.metrics,
			EarlyStopping(monitor='val_loss', patience=10),
			ModelCheckpoint(filepath=save_fname, 
            monitor='val_loss', save_best_only=True), 
            tensorboard
		]

        self.model.add(LSTM(20, activation= 'relu', return_sequences=True,
        input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(100, activation= 'relu'))
        # self.model.add(Dense(10, activation= 'relu'))
        self.model.add(BatchNormalization())
        # self.model.add(Dense(20, activation= 'relu'))
        self.model.add(Dropout(0.62))
        #self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary()

        adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
        self.model.compile(optimizer= adam,
            loss=  'binary_crossentropy', 
            metrics=['accuracy']
            )

        history = self.model.fit(self.X_train,
                    self.y_train,
                    epochs = 6,
                    batch_size=256,
                    #class_weight = self.class_weights, 
                    validation_data = (self.X_val, self.y_val), 
                    callbacks = callbacks, 
                    verbose=1)

        self.model.save(save_fname)

        return history 

    def predict(self, X_test, model_file):
        self.load_model(model_file)

        return self.model.predict(X_test)

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        rr = (np.intersect1d(val_targ, val_predict))
        if len(rr) == 0:
            print('no prediction equal to 1')
        _val_precision = np.float(len(rr)) / (len(val_predict) + K.epsilon())
        _val_recall = np.float(len(rr)) / (len(val_targ) + K.epsilon())
        _val_f1 = 2 * _val_precision*_val_recall / (_val_precision + _val_recall + K.epsilon())
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print ('—val_f1: %f—val_precision: %f—val_recall %f' %(_val_f1, _val_precision, _val_recall))
        return

class PlotAccuracy:
    def plot(self, history, n_epoch):
        history_dict = history.history
        acc_values = history_dict['acc']
        val_acc_values = history_dict['val_acc']
        epochs = range(1, n_epoch+1)
        plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.show()


if __name__ == '__main__':
    from tabulate import tabulate
    import pandas as pd

    from dao.load_data import DoubleStrategyLoadData as train
    from dao.load_depth import GetDepth

    from dao.constant import EX_TRANS_FEE, HUOBI, BINANCE
    from getXY.get_XY_depth import DataPrepareForXY as create_XY 
    from forecast.NN import Model, PlotAccuracy
    ####################
    ## Load Data
    pd_kline = train().coin_kline(
        coin = 'xrp', 
        base_currency = 'usdt', 
        start = '1 December 2018 00:00', 
        end = '31 January 2019 00:00', 
        exchange=BINANCE)
    print(tabulate(pd_kline.head(5), headers = 'keys', tablefmt="psql"))

    pd_depth = GetDepth().load_depth(
        exchange = BINANCE, 
        coin = 'xrpusdt',
        start = '1 December 2018 00:00', 
        end = '31 January 2019 00:00', 
    )
    print(tabulate(pd_depth.head(5), headers = 'keys', tablefmt="psql"))

    pd_kd = pd.concat([
        pd_kline, 
        pd_depth],
        axis = 1, 
        join = 'inner')
    print(tabulate(pd_kd.head(), headers = 'keys', tablefmt="psql"))


    ##########################
    ### Create X Y
    X, Y, feature_names, price, date_minute =  create_XY(
        lookback_minutes=60, lookforward_minutes=10).get_XY(
        data_original = pd_kd, 
        up_factor = 1.0, 
        down_factor= 0.5 
        )

    X_train, X_val, X_test, Y_train, Y_val, Y_test = create_XY(
        lookback_minutes=60, lookforward_minutes=10).train_test_split(
            X,
            Y)

    print(len(Y_test))
    print(len(Y_train[Y_train == 1]))
    print(len(Y_test[Y_test == 1]))
    len(Y_test[Y_test == 1])/len(Y_test)

    #################################
    #### Run Fit 
    history = Model(X_train, Y_train, X_val, Y_val).lstm_fit()
