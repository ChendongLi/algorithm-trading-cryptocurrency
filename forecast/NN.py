'''.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:
@author CL
@email lichendonger@gmail.com
@copyright CL all rights reserved
@created Thu Dec 22 2018 16:15 GMT-0800 (PST)
@last-modified Tue Feb 26 2019 18:25 GMT-0800 (PST)
.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:'''

import logging
import utils.logsetup
import numpy as np
# import matplotlib.pyplot as plt
import os
import datetime as dt
import json

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from keras.layers import Dense, Activation, BatchNormalization, Dropout, LSTM, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras import backend as K
from keras import optimizers
from keras.callbacks import Callback
from sklearn.utils import class_weight

logger = logging.getLogger(__name__)

class Model:
    '''
    deep learning model to train and predict 
    '''
    def __init__(self, X_train, y_train, X_val, y_val):
        K.clear_session()
        self.configs = json.load(open('config.json', 'r'))
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
        '''
        load_model: load already-trained model
        Params:
                filepath (str)
        Return:
                model  (obj, h5)
        '''
        logger.info('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def f1_loss(self, y_true, y_pred):
        '''
        f1_loss: customized loss function to minimize F1 score
        Params: 
                y_true (array): true target value
                y_predict (array): predict target value
        Return: 
                return (array):  minimize loss function
        '''
        tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
        tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2*p*r / (p+r+K.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        return 1 - K.mean(f1)

    def nn_fit(self, save_fname):
        '''
        nn_fit: train multiple perception neural network model
        Params: 
                save_fname (str): trained h5 model url 
        Return: history (obj, h5): training history result
        '''

        NAME = 'Feedforward' 
        logger.info(f'{NAME} Neural Network Training Started')

        tensorboard = TensorBoard(log_dir=f'logs-{NAME}')
        callbacks = [
            self.metrics,
			EarlyStopping(monitor='val_loss', 
                patience=self.configs['nn']['early_stop_patience']),
			ModelCheckpoint(filepath=save_fname, 
            monitor='val_loss', save_best_only=True),
            tensorboard
		    ]

        # self.model.add(Dense(100, activation= 'relu', 
        # input_shape=(self.X_train.shape[1], )))
        # self.model.add(Dense(10, activation= 'relu'))
        # self.model.add(Dense(20, activation= 'relu'))
        # self.model.add(Dropout(0.74))
        # self.model.add(Dense(1, activation='sigmoid'))
        # self.model.summary()
        # self.model.compile(optimizer='adam',
        #     loss=  'binary_crossentropy', 
        #     metrics=['accuracy']
        #     )

        for layer in self.configs['nn']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None

            if layer['type'] == 'layer_input':
                self.model.add(Dense(neurons, activation=activation,
                input_shape=(self.X_train.shape[1], )))
            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))
            if layer['type'] == 'layer_output':
                self.model.add(Dense(neurons, activation=activation))
        
        self.model.summary()
        
        self.model.compile(loss=self.configs['nn']['loss'], 
                            optimizer=self.configs['nn']['optimizer'], 
                            metrics=[self.configs['nn']['accuracy']])
            
        history = self.model.fit(self.X_train,
                    self.y_train,
                    epochs=self.configs['nn']['epochs'],
                    batch_size=self.configs['nn']['batch_size'],
                    class_weight = self.class_weights, 
                    validation_data = (self.X_val, self.y_val), 
                    callbacks = callbacks, 
                    verbose=1)

        self.model.save(save_fname)

        return history

    def lstm_fit(self, save_fname):
        '''
        nn_fit: train recurrent LSTM neural network model
        Params: 
                save_fname (str): trained h5 model url 
        Return: history (obj, h5): training history result
        '''

        NAME = 'LSTM' 
        print(f'{NAME} Neural Network Training Started')
        save_fname = os.path.join(
            'saved_models', 
            '%s-%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'),str('lstm'))
            )

        tensorboard = TensorBoard(log_dir=f'logs-{NAME}')
        callbacks = [
            self.metrics,
			EarlyStopping(monitor='val_loss', 
                            patience=self.configs['lstm']['early_stop_patience']),
			ModelCheckpoint(filepath=save_fname, 
            monitor='val_loss', save_best_only=True), 
            tensorboard
		]

        # self.model.add(LSTM(20, activation= 'relu', return_sequences=True,
        # input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        # self.model.add(BatchNormalization())
        # self.model.add(LSTM(100, activation= 'relu'))
        # # self.model.add(Dense(10, activation= 'relu'))
        # self.model.add(BatchNormalization())
        # # self.model.add(Dense(20, activation= 'relu'))
        # self.model.add(Dropout(0.62))
        # #self.model.add(Flatten())
        # self.model.add(Dense(1, activation='sigmoid'))
        # self.model.summary()

        for layer in self.configs['lstm']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None

            if layer['type'] == 'layer_input':
                self.model.add(LSTM(neurons, activation=activation, return_sequences=return_seq, 
                input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
            if layer['type'] == 'layer_bachnorm':
                self.model.add(BatchNormalization())
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, activation=activation))
            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))
            if layer['type'] == 'layer_output':
                self.model.add(Dense(neurons, activation=activation))

        self.model.summary()

        adam = optimizers.Adam(lr=self.configs['lstm']['optimizer_lr'], 
                                beta_1=self.configs['lstm']['optimizer_beta1'], 
                                beta_2=self.configs['lstm']['optimizer_beta2'])
        self.model.compile(optimizer= adam,
            loss=self.configs['lstm']['loss'], 
            metrics=[self.configs['lstm']['accuracy']]
            )

        history = self.model.fit(self.X_train,
                    self.y_train,
                    epochs=self.configs['lstm']['epochs'],
                    batch_size=self.configs['lstm']['batch_size'],
                    class_weight = self.class_weights, 
                    validation_data = (self.X_val, self.y_val), 
                    callbacks = callbacks, 
                    verbose=1)

        self.model.save(save_fname)

        return history 

    def predict(self, X_test, model_file):
        '''
        predict: given X , predict Y
        Params:
                X_test (array)
                model_file (str): file location
        Return:
                Y predict (array)
        '''
        self.load_model(model_file)

        return self.model.predict(X_test)

# class PlotAccuracy:
#     def plot(self, history, n_epoch):
#         history_dict = history.history
#         acc_values = history_dict['acc']
#         val_acc_values = history_dict['val_acc']
#         epochs = range(1, n_epoch+1)
#         plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
#         plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
#         plt.title('Training and validation accuracy')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.legend(loc='best')
#         plt.show()

class Metrics(Callback):
    '''
    Metrics: customize function used for Keras Model Callback
            to show confusion matrix score precision and recall 
    '''
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        rr = (np.intersect1d(val_targ, val_predict))

        if len(rr) == 0:
            logger.info('no prediction equal to 1')

        _val_precision = np.float(len(rr)) / (len(val_predict) + K.epsilon())
        _val_recall = np.float(len(rr)) / (len(val_targ) + K.epsilon())
        _val_f1 = 2 * _val_precision*_val_recall / (_val_precision + _val_recall + K.epsilon())

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)

        logger.info('—val_f1: %f—val_precision: %f—val_recall %f' %(_val_f1, _val_precision, _val_recall))
        return
