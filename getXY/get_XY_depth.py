import datetime
import numpy as np
import pandas as pd
from dao.load_data import DoubleStrategyLoadData as train
from dao.constant import EX_TRANS_FEE, HUOBI, BINANCE, BITMEX, TRAINING_DATA_BATCH_SIZE
from tabulate import tabulate

from indicator.indicator import williams_r
from indicator.indicator import rsi as rsi_indicator
from indicator.indicator import directional_movement_index as adx_indicator

import warnings

class DataPrepareForXY:
    def __init__(self, lookback_minutes, lookforward_minutes):
        self.random_state = 42
        self.n_jobs = 1
        self.train_val = 0.7
        self.train_test = 0.9
        self.lookback_minutes = lookback_minutes 
        self.step = 1
        self.lookforward_minutes = lookforward_minutes
        self.rolling_window = self.lookback_minutes
        self.dir = '/Users/cli/Data/pair_selection/forecast_skewness'

    def remap(self, x):
        x = np.array(x)
        p = int(len(x) * self.train_val)
        x_train = x[0:p]

        return (x - x_train.min()) / (x_train.max() - x_train.min())

    def moving_average_convergence(self, group):
        nslow = 7
        nfast = 21

        emaslow = group.ewm(span=nslow, min_periods=1).mean().values.tolist()
        emafast = group.ewm(span=nfast, min_periods=1).mean().values.tolist()

        return np.array(emafast) -np.array(emaslow)
    
    def shuffle_in_unison(self, a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]

        return shuffled_a, shuffled_b
    
    def train_test_split(self, X, y):
        p_val = int(len(X) * self.train_val)
        p_test = int(len(X) * self.train_test)

        X_train = X[0:p_val]
        Y_train = y[0:p_val]
        X_val = X[p_val:p_test]
        Y_val = y[p_val:p_test]

        X_test = X[p_test:]
        Y_test = y[p_test:]

        return (X_train, X_val, X_test, Y_train, Y_val, Y_test)

    def get_feature_names(self, column_names, lookback_minutes):

        feature_names = []
        for i in range(0, len(column_names)):
            for n in range(lookback_minutes-1, -1, -1):
                feature_names.append(column_names[i]+str(n))
        
        return feature_names

    def get_XY(self, data_original, up_factor, down_factor):

        date_minute = data_original.index.tolist()
        pricep = data_original.loc[:, 'close'].tolist()
        closep = self.remap(data_original.loc[:, 'close'].tolist())
        volumep = self.remap(data_original.loc[:, 'amount'].tolist())   
        abp_cumdiffp = self.remap(data_original.loc[:, 'abp_cumdiff'].tolist())
        abp_spreadp = self.remap(data_original.loc[:, 'abp_spread'].tolist())  
        abs_cumdiffp = self.remap(data_original.loc[:, 'abs_cumdiff'].tolist())
        abs_spreadp = self.remap(data_original.loc[:, 'abs_spread'].tolist())    
        ap_avgp = self.remap(data_original.loc[:, 'ap_avg'].tolist())
        as_sump = self.remap(data_original.loc[:, 'as_sum'].tolist()) 
        bp_avgp = self.remap(data_original.loc[:, 'bp_avg'].tolist())
        bs_sump = self.remap(data_original.loc[:, 'bs_sum'].tolist())
        midpp = self.remap(data_original.loc[:, 'midp'].tolist())
        ibp = self.remap(data_original.loc[:, 'ib'].tolist()) 

        label, X, Y, close_test, dateminute= {}, [], [], [], []
        for i in range(self.lookback_minutes, len(data_original)-self.lookforward_minutes
                        -self.lookback_minutes, self.step): 
            c = closep[i:i+self.lookback_minutes]
            v = volumep[i:i+self.lookback_minutes]
            abp_cumdiff = abp_cumdiffp[i:i+self.lookback_minutes]
            abp_spread = abp_spreadp[i:i+self.lookback_minutes]
            abs_cumdiff =  abs_cumdiffp[i:i+self.lookback_minutes]
            abs_spread = abs_spreadp[i:i+self.lookback_minutes]
            ap_avg = ap_avgp[i:i+self.lookback_minutes]
            as_sum = as_sump[i:i+self.lookback_minutes]
            bp_avg = bp_avgp[i:i+self.lookback_minutes]
            bs_sum = bs_sump[i:i+self.lookback_minutes]
            midp =  midpp[i:i+self.lookback_minutes]
            ib = ibp[i:i+self.lookback_minutes]

            # macd = self.remap(macd)
            # williams = self.remap(williams)
            # relative = self.remap(relative)
            # ichi = self.remap(ichi)
            # adx_index = self.remap(adx_index)
            # adx = self.remap(adx)
            # o = self.remap(o)
            # h = self.remap(h)
            # l = self.remap(l)
            # c = self.remap(c)
            # v = self.remap(v)
            # volat = self.remap(volat)
            # rsk = self.remap(rsk)
            # rku = self.remap(rku)

            ###### if only use price 
            # x_i = np.reshape(c, 
            # (self.lookback_minutes, -1))

            x_i = np.column_stack((
                 c, v, ib, abp_cumdiff, abs_cumdiff, abp_spread, abs_spread))
                #v , abp_cumdiff, abp_spread, abs_cumdiff, abs_spread))
                #, ap_avg, as_sum, bp_avg, bs_sum, midp
                
            column_names = [
                'close', 'v', 'ib', 'abp_cumdiff', 'abs_cumdiff', 'abp_spread', 'abs_spread' ]
                #'v', 'abp_cumdiff', 'abp_spread', 'abs_cumdiff', 'abs_spread']
                #, 'ap_avg', 'as_sum', 'bp_avg', 'bs_sum', 'midp'
                #]
            #x_i = x_i.flatten()

            for j in range(i+self.lookback_minutes
            , i+self.lookback_minutes+self.lookforward_minutes):
                warnings.filterwarnings('error')
                try:
                    label['forward%s' % str(j)] = (
                        closep[j] - closep[i+self.lookback_minutes -1]
                        )/closep[i+self.lookback_minutes -1] * 100
                except ZeroDivisionError as e:
                    label['forward%s' % str(j)] = 0.0
                    print('zero eorr', e)
                except Warning as e:
                    label['forward%s' % str(j)] = 0.0
                    print('zero warning', e)

            warnings.resetwarnings()
            warnings.filterwarnings("ignore",category=DeprecationWarning)
            
            ####################
            ###label trend 
            #     if j == i+ self.lookback_minutes:
            #         down_tot = (label['forward%s' % str(j)] > - down_factor)
            #         up_tot = (label['forward%s' % str(j)] > up_factor)    
            #     else:  
            #         d = (label['forward%s' % str(j)] > - down_factor)
            #         u = (label['forward%s' % str(j)] > up_factor)
                
            #         down_tot = down_tot & d 
            #         up_tot = up_tot | u
            
            # label['UpDown'] = down_tot & up_tot

            #####################
            ### baseline test for single up or down
            # if label['forward%s' % str(i+self.lookback_minutes)] > 0:
            #     label['UpDown'] = 1
            # else:
            #     label['UpDown']= 0

            ######################
            ## label spike
            if label['forward%s' % str(i+self.lookback_minutes)] <- down_factor :
                label['UpDown'] = 1
            else:
                label['UpDown']= 0       
            
            y_i = int(label['UpDown'])
            closeptest_i = pricep[i+self.lookback_minutes
                            ]
            dateminute_i = date_minute[i+self.lookback_minutes
                          ]

            X.append(x_i)
            Y.append(y_i)
            close_test.append(closeptest_i)
            dateminute.append(dateminute_i)

        X, Y = np.array(X), np.array(Y)

        feature_names = self.get_feature_names(column_names, self.lookback_minutes)

        return X, Y, feature_names, close_test, dateminute