import datetime
import numpy as np
import pandas as pd
from tabulate import tabulate

from indicator.indicator import williams_r
from indicator.indicator import rsi as rsi_indicator
from indicator.indicator import directional_movement_index as adx_indicator

import warnings

class DataPrepareForXY:
    '''
    DataPrepareForXY: process data to get input X and taget Y
    '''
    def __init__(self, train_val, train_test):
        '''
        Params:
                train_val (float): train and validation split 
                train_test (float): train and test split
        '''
        self.train_val = train_val
        self.train_test = train_test

    def remap(self, x):
        '''
        remap: normalize x between -1 and 1
        Paras: 
                x (array): input X
        Return: 
                (array): normalized input X
        '''
        x = np.array(x)
        p = int(len(x) * self.train_val)
        x_train = x[0:p]

        return (x - x_train.min()) / (x_train.max() - x_train.min())
        
    def train_test_split(self, X, y):
        '''
        train_test_split: split train and test into train, validation, test set
        Paras:
                X (array): input X 
                y (array): target y
        Return: 
                (array): train, valation, test for input X and target Y
        '''
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
        '''
        get_feature_names: get each feature name to evalute feature importance
        Paras:
                column_names (str)
                lookback_minutes (int)
        Return: 
                feature_names (list)
        '''
        feature_names = []
        for i in range(0, len(column_names)):
            for n in range(lookback_minutes-1, -1, -1):
                feature_names.append(column_names[i]+str(n))
        
        return feature_names

    def get_XY(self, data_original, lookback_minutes, lookforward_minutes, 
                up_factor, down_factor, step, model, label_category):
        '''
        get_XY: data process to get input X and target Y
        Params:
                data_original (obj pandas dataframe)
                lookback_minute (int): the number of historical minutes to predict future
                lookforward_minutes (int): the number of future minutes for prediction 
                up_factor (float): price increase percentage (i.e 1.0 is up 1.0%)
                down_factor (float): price decrease percentage (i.e 1.0 is down 1.0%)
                step (int): input interval (i.e. 1 minute, 2 minutes etc)
        Return: 
                X (array): input X
                Y (array): taget Y
                feature_name (string)
                closet_test (array): original price in test set
                dateminute (array): original dataminute through all input X
        '''
        n = 0
        date_minute = data_original.index.tolist()
        pricep = data_original.loc[:, 'close'].tolist()

        #### Input Feature
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
        for i in range(lookback_minutes, len(data_original)-lookforward_minutes
                        -lookback_minutes, step): 
            c = closep[i:i+lookback_minutes]
            v = volumep[i:i+lookback_minutes]
            abp_cumdiff = abp_cumdiffp[i:i+lookback_minutes]
            abp_spread = abp_spreadp[i:i+lookback_minutes]
            abs_cumdiff =  abs_cumdiffp[i:i+lookback_minutes]
            abs_spread = abs_spreadp[i:i+lookback_minutes]
            ap_avg = ap_avgp[i:i+lookback_minutes]
            as_sum = as_sump[i:i+lookback_minutes]
            bp_avg = bp_avgp[i:i+lookback_minutes]
            bs_sum = bs_sump[i:i+lookback_minutes]
            midp =  midpp[i:i+lookback_minutes]
            ib = ibp[i:i+lookback_minutes]

            x_i = np.column_stack((
                 c, v, ib, abp_cumdiff, abs_cumdiff, abp_spread, abs_spread))
                
            column_names = [
                'close', 'v', 'ib', 'abp_cumdiff', 'abs_cumdiff', 'abp_spread', 'abs_spread' ]

            if model == 'nn':
                x_i = x_i.flatten()

            for j in range(i+lookback_minutes
            , i+lookback_minutes+lookforward_minutes):
                warnings.filterwarnings('error')

                ### try if divide zero
                try:
                    label['forward%s' % str(j)] = (
                        closep[j] - closep[i+lookback_minutes -1]
                        )/closep[i+lookback_minutes -1] * 100
                except ZeroDivisionError as e:
                    label['forward%s' % str(j)] = 0.0
                    print('zero eorr', e)
                except Warning as e:
                    label['forward%s' % str(j)] = 0.0
                    print('zero warning', e)

            warnings.resetwarnings()
            warnings.filterwarnings("ignore",category=DeprecationWarning)
            
            #####################
            ### baseline test for single up or down
            if label_category == 'updown':
                if label['forward%s' % str(i+lookback_minutes)] > 0:
                    label['UpDown'] = 1
                else:
                    label['UpDown']= 0

            ######################
            ## label spike
            if label_category == 'spike':
                if label['forward%s' % str(i+lookback_minutes)] <- down_factor :
                    label['UpDown'] = 1
                else:
                    label['UpDown']= 0       
            
            y_i = int(label['UpDown'])
            closeptest_i = pricep[i+lookback_minutes
                            ]
            dateminute_i = date_minute[i+lookback_minutes
                          ]

            X.append(x_i)
            Y.append(y_i)
            close_test.append(closeptest_i)
            dateminute.append(dateminute_i)

        X, Y = np.array(X), np.array(Y)

        feature_names = self.get_feature_names(column_names, lookback_minutes)

        return X, Y, feature_names, close_test, dateminute