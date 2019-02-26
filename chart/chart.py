from tabulate import tabulate
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import seaborn.apionly as sns
sns.set_style("white")
sns.set_context("poster")
import pandas as pd 
import mpld3


def chart(data):
        mpld3.enable_notebook()
        
        fig, ax = plt.subplots(2,1, figsize = (20, 8));
        ax[0].plot(data['close'], color='b', label = 'mid_price', alpha = 0.5)
        ax[1].plot(data['abp_cumdiff'], color='g', label = 'imbalance', alpha = 0.5)
        ax[0].legend(loc = 'best')
        ax[1].legend(loc = 'best')
        #ax[0].grid(color = 'white',  linestyle = 'solid')
        # dim = len(data['amount'])
        # w = 0.75
        # dimw = w / dim

        # x = np.arange(len(data['amount']))
        # for i in range(len(data['amount'])) :
        #     y = data.loc[data.index[i], 'amount']
        #     b = ax[1].bar(x + i * dimw, y, dimw, bottom=0.001)
        
        #     ax[1].plot(data['amount'], label = 'amount', alpha = 0.5)
        #     ax[1].plot(data['amount'].loc[data['UpDown'] == 1],
        #             '^', markersize=10, color='g',
        #             label = 'up', alpha = 0.5)
        #     ax[1].legend(loc = 'best')

        mpld3.show()