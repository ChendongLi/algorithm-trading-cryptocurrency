#%%
%load_ext autoreload
%autoreload 2

from tabulate import tabulate
import datetime
import pandas as pd 
import numpy as np 

from dao.load_data import DoubleStrategyLoadData as train
from dao.load_depth import GetDepth
from chart.chart import chart

from dao.constant import EX_TRANS_FEE, HUOBI, BINANCE
from getXY.get_XY_depth import DataPrepareForXY as create_XY 
from forecast.NN import Model, PlotAccuracy

#%%
####################
## Load Data
pd_kline = train().coin_kline(
    coin = 'xrp', 
    base_currency = 'usdt', 
    start = '1 November 2018 00:00', 
    end = '18 February 2019 00:00', 
    exchange=BINANCE)
print(tabulate(pd_kline.head(5), headers = 'keys', tablefmt="psql"))

pd_depth = GetDepth().load_depth(
    exchange = BINANCE, 
    coin = 'xrpusdt',
    start = '1 November 2018 00:00', 
    end = '18 February 2019 00:00', 
)
print(tabulate(pd_depth.head(5), headers = 'keys', tablefmt="psql"))

pd_kd = pd.concat([
    pd_kline, 
    pd_depth],
    axis = 1, 
    join = 'inner')
print(tabulate(pd_kd.head(), headers = 'keys', tablefmt="psql"))


#%%
##########################
### Create X Y
X, Y, feature_names, price, date_minute =  create_XY(
    lookback_minutes=60, lookforward_minutes=20).get_XY(
    data_original = pd_kd, 
    up_factor = 1.0, 
    down_factor= 1.0 
    )

#%%
X_train, X_val, X_test, Y_train, Y_val, Y_test = create_XY(
    lookback_minutes=60, lookforward_minutes=20).train_test_split(
        X,
        Y)
print(len(Y_train))
print(len(Y_train[Y_train == 1]))
print('train', len(Y_train[Y_train == 1])/len(Y_train))
print(len(Y_val[Y_val == 1]))
len(Y_val[Y_val == 1])/len(Y_val)
#%%
history = Model(X_train, Y_train, X_val, Y_val).lstm_fit()

#%%
#Y_predict = nn_model(X_train, Y_train, X_test, Y_test).predict(X_test)

Y_predict = Model(X_train, Y_train, X_val, Y_val).predict(
        X_val, 
        './saved_models/18022019-215629-lstm.h5')
#%%
def performanec_metric(Y_predict, Y_test, prob_limit):
    temp =pd.DataFrame(np.column_stack(
        (
        Y_predict, 
        Y_val
        )
    ))
    temp.columns = ['Predict', 'Test']

    print(temp['Predict'].describe())

    temp_filter = temp[(temp['Predict'] >= prob_limit) & (temp['Test'] == 1)]
    _recall = len(temp_filter)/len(Y_test[Y_test == 1])
    _precision = len(temp_filter)/len(temp[temp['Predict'] >= prob_limit])
    _f1_score = 2 * _recall * _precision/(_recall + _precision)
    print('recall', _recall)
    print('precision', _precision)
    print('f1 score', _f1_score)

performanec_metric(Y_predict, Y_test, prob_limit = 0.2)

#%%
from Tune_NN import *

X_train, Y_train, X_test, Y_test = data(
    coin = 'xrp', 
    base_currency = 'btc', 
    exchange = BINANCE, 
    start = '1 December 2018 00:00', 
    end = '10 December 2018 00:00', 
    lookback_minutes = 60,
    lookforward_minutes = 1,
    up_factor=0.3, 
    down_factor=0.2)

#%%
temp =pd.DataFrame(np.column_stack(
    (
    price_test, 
    Y_predict, 
    Y_test 
    )
), index = date_minute[p:])
temp.columns = ['Price', 'Predict', 'Test']


#%%
X_train.shape

#%%
