'''.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:
@author CL
@email lichendonger@gmail.com
@copyright CL all rights reserved
@created Thu Dec 23 2018 15:10 GMT-0800 (PST)
@last-modified Wed Feb 27 2019 19:20 GMT-0800 (PST)
.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:'''

from tabulate import tabulate
import matplotlib
import os
import datetime
from matplotlib import pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import numpy as np
import seaborn.apionly as sns
sns.set_style("white")
sns.set_context("poster")
import pandas as pd 
import mpld3

class VisualChart:
    '''
    plot visualization
    '''
    def price_volumn(self, kline_data, save_figure):
        '''
        price_volumn: plot price and volumn 
        Param:
            kline_data (obj, pandas_dataframe)
        '''
        fig_file = os.path.join(save_figure, 
                '%s-%s.png' % (('price-volumn-'), datetime.datetime.now().strftime('%Y%m%d-%H%M')))

        fig, ax = plt.subplots(1, 1, figsize = (12, 8))
        ax2 = ax.twinx()
        ax.set_ylabel('Price($)')
        ax2.set_ylabel('Volume')
        ax.set_title('Ripple (XRP) Market Price')
        kline_data['date'] = kline_data.index.map(mdates.date2num)

        ohlc = kline_data[['date', 'open', 'high', 'low', 'close']]
        candlestick_ohlc(ax, ohlc.values.tolist(), width=.005, colorup='#77d879', colordown='#db3f3f', alpha = 0.5)

        ax2.fill_between(kline_data['date'].tolist(),0, kline_data['vol'].tolist(), facecolor='#0079a3', alpha=0.7)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(fig_file)
        #mpld3.show()
        #plt.show()

    def model_evaluation(self, history, n_epoch, save_figure):
        '''
        model_evaluation: plot training and volidation accuracy 
        Params:
                hisotry (obj): training model history 
                n_epoch (int): number of epoch 
        '''
        fig_file = os.path.join(save_figure, 
                '%s-%s.png' % (('model-evaluation-'), datetime.datetime.now().strftime('%Y%m%d-%H%M')))

        fig, ax = plt.subplots(1, 1, figsize = (12, 8))
        history_dict = history.history
        acc_values = history_dict['acc']
        val_acc_values = history_dict['val_acc']
        epochs = range(1, n_epoch+1, 1)
        ax.plot(epochs, acc_values, 'bo', color = 'blue', label='Training accuracy')
        ax.plot(epochs, val_acc_values, 'b', color = 'orange', label='Validation accuracy')
        ax.set_xticks(epochs)
        ax.set_title('Training and validation accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend(loc='best')
        fig.savefig(fig_file)
        #mpld3.show()
        #plt.show()

    # def chart(data):
    #     mpld3.enable_notebook()
        
    #     fig, ax = plt.subplots(2,1, figsize = (20, 8));
    #     ax[0].plot(data['close'], color='b', label = 'mid_price', alpha = 0.5)
    #     ax[1].plot(data['abp_cumdiff'], color='g', label = 'imbalance', alpha = 0.5)
    #     ax[0].legend(loc = 'best')
    #     ax[1].legend(loc = 'best')
    #     #ax[0].grid(color = 'white',  linestyle = 'solid')
    #     # dim = len(data['amount'])
    #     # w = 0.75
    #     # dimw = w / dim

    #     # x = np.arange(len(data['amount']))
    #     # for i in range(len(data['amount'])) :
    #     #     y = data.loc[data.index[i], 'amount']
    #     #     b = ax[1].bar(x + i * dimw, y, dimw, bottom=0.001)
        
    #     #     ax[1].plot(data['amount'], label = 'amount', alpha = 0.5)
    #     #     ax[1].plot(data['amount'].loc[data['UpDown'] == 1],
    #     #             '^', markersize=10, color='g',
    #     #             label = 'up', alpha = 0.5)
    #     #     ax[1].legend(loc = 'best')

        #mpld3.show()