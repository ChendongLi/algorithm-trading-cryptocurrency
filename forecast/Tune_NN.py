import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import datetime as dt
from tabulate import tabulate

import tensorflow as tf
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

from dao.load_data import DoubleStrategyLoadData as train
from dao.load_depth import GetDepth

from getXY.get_XY_depth import DataPrepareForXY as create_XY 

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        rr = (np.intersect1d(val_targ, val_predict))
        _val_precision = np.float(len(rr)) / (len(val_predict) + K.epsilon())
        _val_recall = np.float(len(rr)) / (len(val_targ) + K.epsilon())
        _val_f1 = 2 * _val_precision*_val_recall / (_val_precision + _val_recall + K.epsilon())
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print ('—val_f1: %f—val_precision: %f—val_recall %f' %(_val_f1, _val_precision, _val_recall))
        return

def data():

    coin = 'xrp'
    base_currency = 'btc'
    exchange = BINANCE
    start = '28 January 2019 00:00'
    end = '31 January 2019 00:00' 
    lookback_minutes = 60
    lookforward_minutes = 1
    up_factor=0.3
    down_factor=0.2

    pd_kline = train().coin_kline(
    coin = coin, 
    base_currency = base_currency, 
    start = start, 
    end = end,  
    exchange= exchange)
    #print(tabulate(pd_kline.head(5), headers = 'keys', tablefmt="psql"))

    pd_depth = GetDepth().load_depth(
        exchange = exchange, 
        coin = coin, 
        base_currency = base_currency, 
        start = start, 
        end = end
    )
    #print(tabulate(pd_depth.head(5), headers = 'keys', tablefmt="psql"))

    pd_kd = pd.concat([
        pd_kline, 
        pd_depth],
        axis = 1, 
        join = 'inner')
    #print(tabulate(pd_kd.head(), headers = 'keys', tablefmt="psql"))

    X, Y, feature_names, price, date_minute =  create_XY(
        lookback_minutes=lookback_minutes, 
        lookforward_minutes=lookforward_minutes).get_XY(
        data_original = pd_kd, 
        up_factor = up_factor, 
        down_factor= down_factor 
        )

    X_train, X_val, X_test, Y_train, Y_val, Y_test = create_XY(
        lookback_minutes=lookback_minutes, 
        lookforward_minutes=lookforward_minutes).train_test_split(
            X,
            Y)
    
    return X_train, Y_train, X_val, Y_val

def create_model(X_train, Y_train, X_val, Y_val):
    print('Tune LSTM Started')
    save_fname = os.path.join(
        'saved_models', 
        '%s-%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'),str('lstm'))
        )

    callbacks = [
        #metrics,
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint(filepath=save_fname, 
        monitor='val_loss', save_best_only=True)
    ]

    model = Sequential()
    model.add(LSTM({{choice([10, 20, 50,  100, 150, 200])}}, return_sequences=True, activation= 'relu', 
    input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(BatchNormalization())
    model.add(LSTM({{choice([10, 20, 50,  100, 150, 200])}}, activation= 'relu'))
    model.add(BatchNormalization())
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense(20, activation= 'relu'))
    # model.add(Dense(20, activation= 'relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    #self.model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.add(BatchNormalization())
    #model.summary()
    adam = optimizers.Adam(lr={{choice([0.001, 0.0001, 0.00001])}}, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer= adam, #{{choice(['rmsprop','adam', 'sgd'])}}, 
        loss=  'binary_crossentropy', 
        metrics=['accuracy']
        )

    history = model.fit(
        X_train,
        Y_train,
        epochs = 5,
        batch_size= {{choice([16,32,64,128,256])}}, 
        ##class_weight = self.class_weights, 
        validation_data = (X_val, Y_val), 
        # callbacks = callbacks, 
        verbose=1)

    # model.save(save_fname)

    validation_acc = np.amax(history.history['val_acc']) 
    
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

    X_train, Y_train, X_val, Y_val = data()

    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_val, Y_val))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)