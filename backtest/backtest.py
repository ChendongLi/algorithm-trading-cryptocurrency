#%%
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt
from tabulate import tabulate

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

from forecast.NN import Model
from dao.load_data import DoubleStrategyLoadData as train
from dao.load_depth import GetDepth
from getXY.get_XY_depth import DataPrepareForXY as create_XY 

def get_testXY(coin, base_currency, start, end, exchange, model_file):
    ####################
    ## Load Data
    pd_kline = train().coin_kline(
        coin = coin, 
        base_currency = base_currency, 
        start = start, 
        end = end, 
        exchange=BINANCE)
    print(tabulate(pd_kline.head(5), headers = 'keys', tablefmt="psql"))

    pd_depth = GetDepth().load_depth(
        exchange = BINANCE, 
        coin = coin, 
        base_currency = base_currency, 
        start = start, 
        end = end, 
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
        lookback_minutes=60, lookforward_minutes=20).get_XY(
        data_original = pd_kd, 
        up_factor = 1.0, 
        down_factor= 0.2 
        )

    X_train, X_val, X_test, Y_train, Y_val, Y_test = create_XY(
        lookback_minutes=60, lookforward_minutes=20).train_test_split(
            X,
            Y)

    print(len(X))
    print(len(Y[Y == 1]))
    len(Y[Y == 1])/len(Y)

    #################################
    #### Run Fit 
    Y_predict = Model(X_train, Y_train, X_val, Y_val).predict(X_test, model_file)

    return(Y_test, Y_predict, price[-len(Y_test):], date_minute[-len(Y_test):])


#%%
Y, Y_predict, price, date_minute = get_testXY(
    coin = 'xrp',  
    base_currency = 'usdt', 
    start = '1 November 2018 00:00',
    end = '18 February 2019 00:00',
    exchange = BINANCE, 
    model_file = './saved_models/18022019-215629-lstm.h5'
    )

#%%

def get_tradeData(price, Y_predict, Y, date_minute, buy_limit, sell_limit):

    pd_trade =pd.DataFrame(np.column_stack(
        (
        price, 
        Y_predict, 
        Y
        )
    ), index = date_minute)
    pd_trade.columns = ['price', 'predict', 'test']

    print(pd_trade['predict'].describe())

    def classified_func(row):
        if row['predict'] > sell_limit:
            return -1.0
        elif row['predict'] < buy_limit:
            return 1.0  
        else:
            return 0

    pd_trade['trade'] = pd_trade.apply(classified_func, axis =1 )

    return pd_trade

pd_trade = get_tradeData(price, Y_predict, Y, date_minute, buy_limit = 0.01, sell_limit = 0.20)

#%%
n = pd_trade.index.min() 
pd_trade['decision_type'] = 'nil'
pd_trade['amount'] = 0.0
pd_trade['pnL'] = 0.0
orig_base_total = 0.5
trans_fee = 0.0000 #EX_TRANS_FEE['binance']

for row in pd_trade.itertuples():

    if row.trade == 1.0 and pd_trade.loc[n,'decision_type'] != 'buy_trigger' :
        pd_trade.at[row.Index, 'amount'] = orig_base_total/pd_trade.loc[row.Index, 'price']
        pd_trade.at[row.Index, 'decision_type'] = 'buy_trigger'
        print(row.Index, pd_trade.loc[row.Index, 'price'], 
        pd_trade.loc[row.Index, 'decision_type'])
        n = row.Index

    elif row.predict >= 0.1 and pd_trade.loc[n,'decision_type'] == 'buy_trigger' :
        # if row.predict <= 0.5:
        pd_trade.at[row.Index, 'amount'] = -1.0 * pd_trade.at[n, 'amount']
        pd_trade.at[row.Index, 'decision_type'] = 'buy_taker'
        pd_trade.at[row.Index, 'pnL'] = pd_trade.loc[n, 'amount'] * (
            pd_trade.loc[row.Index, 'price'] - pd_trade.loc[n, 'price']) - trans_fee*(
            abs(pd_trade.loc[n, 'amount']) * pd_trade.loc[n, 'price'] + 
            abs(pd_trade.loc[n, 'amount']) * pd_trade.loc[row.Index, 'price']) 

        print(row.Index, pd_trade.loc[row.Index, 'price'], 
        pd_trade.loc[row.Index, 'decision_type'], 
        100*pd_trade.loc[row.Index, 'pnL']/(pd_trade.loc[row.Index, 'price']* abs(pd_trade.loc[row.Index, 'amount']))
        )
        n = row.Index

    if row.trade == -1.0 and pd_trade.loc[n,'decision_type'] != 'sell_trigger':
        pd_trade.at[row.Index, 'amount'] = -orig_base_total/pd_trade.loc[row.Index, 'price']
        pd_trade.at[row.Index, 'decision_type'] = 'sell_trigger'
        print(row.Index, pd_trade.loc[row.Index, 'price'], 
        pd_trade.loc[row.Index, 'decision_type'])
        n = row.Index

    elif row.predict <= 0.1 and pd_trade.loc[n,'decision_type'] == 'sell_trigger' :
        # if row.predict >= 0.5:
        pd_trade.at[row.Index, 'amount'] = -1.0 * pd_trade.at[n, 'amount']
        pd_trade.at[row.Index, 'decision_type'] = 'sell_taker'
        pd_trade.at[row.Index, 'pnL'] = pd_trade.loc[n, 'amount'] * (
            pd_trade.loc[row.Index, 'price'] - pd_trade.loc[n, 'price']) - trans_fee*(
            abs(pd_trade.loc[n, 'amount']) * pd_trade.loc[n, 'price'] + 
            abs(pd_trade.loc[n, 'amount']) * pd_trade.loc[row.Index, 'price'])
        print(row.Index, pd_trade.loc[row.Index, 'price'], 
        pd_trade.loc[row.Index, 'decision_type'], 
        100*pd_trade.loc[row.Index, 'pnL']/(pd_trade.loc[row.Index, 'price']* abs(pd_trade.loc[row.Index, 'amount']))
        )
        n = row.Index

pd_trade['profit'] = pd_trade['pnL'].cumsum()/orig_base_total * 100

#%%
from matplotlib import pyplot as plt
import numpy as np
import seaborn.apionly as sns
sns.set_style("white")
sns.set_context("poster")
import mpld3

fig, ax = plt.subplots(1, 1, figsize=(12, 15))

ax.plot(pd_trade['price'], color = 'blue', alpha = 0.8, label = 'price')
ax.plot(
    pd_trade['price'].loc[pd_trade['amount'] == -1.0],
    '^',
    markersize=10,
    color='g',
    label='Sell')
ax.legend(loc='best')
# plt.savefig(dir)
mpld3.show()

#%%
pd_trade

#%%
pd_trade['profit'].plot()

#%%
from matplotlib import pyplot
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(Y, Y_predict)

print(roc_auc_score(Y, Y_predict))
pyplot.plot(fpr, tpr)

#%%
pd_trade['price'].plot()

#%%
len(pd_trade[pd_trade['pnL'] > 0]) / len(pd_trade[pd_trade['pnL'] != 0])

#%%
