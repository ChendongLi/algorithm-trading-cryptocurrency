import datetime
import numpy as np
import pandas as pd
from dao.load_data import DoubleStrategyLoadData as train
from dao.constant import EX_TRANS_FEE, HUOBI, BINANCE, BITMEX, TRAINING_DATA_BATCH_SIZE
from tabulate import tabulate

from indicator.indicator import williams_r
from indicator.indicator import rsi as rsi_indicator
from indicator.indicator import directional_movement_index as adx_indicator

class DataPrepareForXY:
    def __init__(self, lookback_minutes, lookforward_minutes):
        self.random_state = 42
        self.n_jobs = 1
        self.test_set = 0.7
        self.lookback_minutes = lookback_minutes 
        self.step = 1
        self.lookforward_minutes = lookforward_minutes
        self.rolling_window = self.lookback_minutes
        self.dir = '/Users/cli/Data/pair_selection/forecast_skewness'

    def remap(self, x):
        x = np.array(x)
        return (x - x.min()) / (x.max() - x.min())

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
        p = int(len(X) * self.test_set)

        X_train = X[0:p]
        Y_train = y[0:p]
        X_test = X[p:]
        Y_test = y[p:]

        return X_train, X_test, Y_train, Y_test

    def get_feature_names(self, column_names, lookback_minutes):

        feature_names = []
        for i in range(0, len(column_names)):
            for n in range(lookback_minutes-1, -1, -1):
                feature_names.append(column_names[i]+str(n))
        
        return feature_names

    def get_XY(self, data_original, up_factor, down_factor):

        date_minute = data_original.index.tolist()
        closep = data_original.loc[:, 'close'].tolist()
        volumep = data_original.loc[:, 'amount'].tolist()   
        abp_cumdiffp = data_original.loc[:, 'abp_cumdiff'].tolist()
        abp_spreadp = data_original.loc[:, 'abp_spread'].tolist()  
        abs_cumdiffp = data_original.loc[:, 'abs_cumdiff'].tolist()
        abs_spreadp = data_original.loc[:, 'abs_spread'].tolist()     
        ap_avgp = data_original.loc[:, 'ap_avg'].tolist()
        as_sump = data_original.loc[:, 'as_sum'].tolist()  
        bp_avgp = data_original.loc[:, 'bp_avg'].tolist()
        bs_sump = data_original.loc[:, 'bs_sum'].tolist()  
        midpp = data_original.loc[:, 'midp'].tolist()
        ibp = data_original.loc[:, 'ib'].tolist()  

        label, X, Y, close_test, dateminute= {}, [], [], [], []
        for i in range(self.lookback_minutes, len(data_original)-self.lookforward_minutes
                        -self.lookback_minutes, self.step): 
            c = self.remap(closep[i:i+self.lookback_minutes])
            v = self.remap(volumep[i:i+self.lookback_minutes])
            abp_cumdiff = self.remap(abp_cumdiffp[i:i+self.lookback_minutes])
            abp_spread = self.remap(abp_spreadp[i:i+self.lookback_minutes])
            abs_cumdiff = self.remap(abs_cumdiffp[i:i+self.lookback_minutes])
            abs_spread = self.remap(abs_spreadp[i:i+self.lookback_minutes])
            ap_avg = self.remap(ap_avgp[i:i+self.lookback_minutes])
            as_sum = self.remap(as_sump[i:i+self.lookback_minutes])
            bp_avg = self.remap(bp_avgp[i:i+self.lookback_minutes])
            bs_sum = self.remap(bs_sump[i:i+self.lookback_minutes])
            midp = self.remap(midpp[i:i+self.lookback_minutes])
            ib = self.remap(ibp[i:i+self.lookback_minutes])

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

            x_i = np.column_stack((
                abp_cumdiff,  abs_cumdiff, ib))
                #v , abp_cumdiff, abp_spread, abs_cumdiff, abs_spread))
                #, ap_avg, as_sum, bp_avg, bs_sum, midp
                
            column_names = [
                'abp_cumdiff', 'abs_cumdiff', 'ib']
                #'v', 'abp_cumdiff', 'abp_spread', 'abs_cumdiff', 'abs_spread']
                #, 'ap_avg', 'as_sum', 'bp_avg', 'bs_sum', 'midp'
                #]
            #x_i = x_i.flatten()

            for j in range(i+self.lookback_minutes
            , i+self.lookback_minutes+self.lookforward_minutes):
                label['forward%s' % str(j)] = (
                    closep[j] - closep[i+self.lookback_minutes -1]
                    )/closep[i+self.lookback_minutes -1] * 100

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
            
            closeptest_i = closep[i+self.lookback_minutes+self.lookforward_minutes
                            ]
            dateminute_i = date_minute[i+self.lookback_minutes+self.lookforward_minutes
                          ]

            X.append(x_i)
            Y.append(y_i)
            close_test.append(closeptest_i)
            dateminute.append(dateminute_i)

        X, Y = np.array(X), np.array(Y)
        #X_train, X_test, Y_train, Y_test = self.create_Xt_Yt(X, Y, self.TEST_SET)
        feature_names = self.get_feature_names(column_names, self.lookback_minutes)

        return X, Y, feature_names, close_test, dateminute