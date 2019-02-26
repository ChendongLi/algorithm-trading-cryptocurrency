#%%
%load_ext autoreload
%autoreload 2

from tabulate import tabulate
from dao.load_data import DoubleStrategyLoadData as train
from dao.load_depth import GetDepth
from chart.chart import chart
import pandas as pd 
from dao.constant import EX_TRANS_FEE, HUOBI, BINANCE
from getXY.get_XY_depth import DataPrepareForXY as create_XY 
from forecast.RandomForestModel import RandomForecastTrendForecasting as rf 
import datetime

#%%
##Load Kline
#############################
pd_kline = train().coin_kline(
    coin = 'xrp', 
    base_currency = 'usdt', 
    start = '1 December 2018 00:00', 
    end = '10 December 2018 00:00', 
    exchange=BINANCE)
print(tabulate(pd_kline.head(10), headers = 'keys', tablefmt="psql"))

pd_kline['close'].plot()
#%%
pd_kline['close_ma'] = pd_kline['close'].rolling('1H').mean()
pd_kline['amount_ma'] = pd_kline['amount'].rolling('1H').mean()

pd_kline['close_pct'] = (pd_kline['close'] 
- pd_kline['close_ma'])/pd_kline['close_ma'] * 100
pd_kline['amount_pct'] = (pd_kline['amount'] 
- pd_kline['amount_ma'])/pd_kline['amount_ma']

pd_kline['high_low'] = (pd_kline['high'] 
- pd_kline['low'])/(pd_kline['close']) * 100
training_start = pd_kline.index.min() + datetime.timedelta(hours=1)
pd_kline = pd_kline[pd_kline.index >= training_start]
pd_kline.drop(columns=['open'], inplace=True)
print(tabulate(pd_kline.head(10), headers = 'keys', tablefmt="psql"))

#%%
#######################################
###EDA Graph
from tabulate import tabulate
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import seaborn.apionly as sns
sns.set_style("white")
sns.set_context("poster")

kline_spike = pd_kline[abs(pd_kline['close_pct']) >= 1]
print(tabulate(kline_spike, headers = 'keys', tablefmt="psql"))

#%%
def eda_chart(data):
    fig, ax = plt.subplots(3,1,figsize = (18, 12));
    ax[0].plot(data['close_pct'], color = 'blue', label = 'price_change' )
    ax[0].legend(loc = 'upper right')

    # ax2 = ax.twinx()
    ax[1].plot(data['amount'], color = 'yellow', label = 'amount_change')
    #ax[1].set_ylim([-50, 50])
    ax[1].legend(loc = 'upper left')

    ax[2].plot(data['close'], color = 'green', label = 'price' )
    ax[2].legend(loc = 'upper right')
eda_chart(kline_spike)


#%%
## Load Depth
#######################################

pd_depth = GetDepth().load_depth(
    exchange = BINANCE, 
    coin = 'xrpusdt',
    start = '1 December 2018 00:00', 
    end = '10 December 2018 00:00', 
)

print(tabulate(pd_depth.head(10), headers = 'keys', tablefmt="psql"))

#%%
pd_kd = pd.concat([
    pd_kline, 
    pd_depth],
    axis = 1, 
    join = 'inner')

print(tabulate(pd_kd.head(), headers = 'keys', tablefmt="psql"))

#%%

pd_spike = pd_kd[abs(pd_kd['close_pct']) >= 1].copy()
pd_spike.drop(
    columns=['high', 'low', 'vol', 'close_ma', 'amount_ma'], 
    inplace = True)
print(tabulate(pd_spike, headers = 'keys', tablefmt="psql"))

# X, Y, feature_names, price, date_minute =  create_XY(
#     lookback_minutes=10, lookforward_minutes=5).get_XY(
#     data_original = pd_kd, 
#     up_factor = 1.0, 
#     down_factor= 0.4 
#     )

# X_train, X_test, y_train, y_test = create_XY(10, 5).train_test_split(X, Y)

## Run Random Forest
#######################################
# for n_estimators in [200, 250]:
#     for max_depth in [3, 4]:
#         print('n_estimator:', n_estimators)
#         print('max_depth:', max_depth)
#         rf().rf_model(
#                 X_train = X_train, 
#                 X_test = X_test, 
#                 y_train = y_train,  
#                 y_test = y_test, 
#                 n_estimators = n_estimators, 
#                 max_depth = max_depth, 
#                 feature_names = feature_names)
# temp = pd.DataFrame(
#     {'close': price, 
#     'label': Y
#     }, index = date_minute 
# )
# temp[temp['label'] == 1]
# chart(pd_kd)

#%%
