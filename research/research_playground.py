
#%%
%load_ext autoreload
%autoreload 2
from dao.load_data import DoubleStrategyLoadData as train
from dao.constant import EX_TRANS_FEE, HUOBI, BINANCE
from tabulate import tabulate
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import seaborn.apionly as sns
sns.set_style("white")
sns.set_context("poster")
import pandas as pd 

#%%

pd_kline = train().coin_kline(
    coin = 'btc'
    , base_currency = 'usdt'
    , start = '1 December 2018'
    , end = '13 January 2019'
    , exchange=BINANCE)


#%%
pd_kline['close_pct'] = pd_kline['close'].pct_change()*100
print(tabulate(pd_kline.head(), headers = 'keys', tablefmt="psql"))

#%%

pd_kline = train().washData(pd_kline, 1000, 'close')


#%%
pd_kline['amount'].describe()

#%%
fig, ax = plt.subplots(1,1, figsize = (20, 8));
ax.plot(pd_kline['close'], label = 'close', alpha = 0.5)
ax2 = ax.twinx()
ax2.plot(pd_kline['amount'], label = 'amount', alpha = 0.5, color = 'brown')
ax.legend(loc = 'best')
ax2.legend(loc = 'best')
plt.show()