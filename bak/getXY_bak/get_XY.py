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
        openp = data_original.loc[:, 'open'].tolist()
        highp = data_original.loc[:, 'high'].tolist()
        lowp = data_original.loc[:, 'low'].tolist()
        closep = data_original.loc[:, 'close'].tolist()
        volumep = data_original.loc[:, 'amount'].tolist()   
        # date_minute = data_original.loc[:, 'date_minute'].tolist()

        # nine_period_high = pd.DataFrame(highp).rolling(window=int(self.rolling_window / 2)).max()
        # nine_period_low = pd.DataFrame(lowp).rolling(window=int(self.rolling_window / 2)).min()
        # ichimoku = (nine_period_high + nine_period_low) /2
        # ichimoku = ichimoku.replace([np.inf, -np.inf], np.nan)
        # ichimoku = ichimoku.fillna(0.).values.tolist()

        # macd_indie = self.moving_average_convergence(pd.DataFrame(closep))

        # wpr = williams_r(data_original, int(self.rolling_window / 2), 'high', 'low', 'close')
        # wpr = wpr.loc[:, 'williams_r'].tolist()
        # rsi = rsi_indicator(data_original, int(self.rolling_window / 2), 'close')
        # rsi = rsi.loc[:, 'rsi'].tolist()
        # adx = adx_indicator(data_original, int(self.rolling_window / 2), 'high', 'low', 'open', 'close')
        # adx = adx.loc[:, 'adx'].tolist()
    
        # volatility = pd.DataFrame(closep).rolling(self.rolling_window).std().values

        # rolling_skewness = pd.DataFrame(closep).rolling(self.rolling_window).skew().values 
        # rolling_kurtosis = pd.DataFrame(closep).rolling(self.rolling_window).kurt().values

        label, X, Y, close_test, dateminute= {}, [], [], [], []
        for i in range(self.lookback_minutes, len(data_original)-self.lookforward_minutes
                        -self.lookback_minutes, self.step): 
            #try:
            o = openp[i:i+self.lookback_minutes]
            h = highp[i:i+self.lookback_minutes]
            l = lowp[i:i+self.lookback_minutes]
            c = closep[i:i+self.lookback_minutes]
            v = volumep[i:i+self.lookback_minutes]
            # volat = volatility[i:i+self.lookback_minutes]
            # rsk = rolling_skewness[i:i+self.lookback_minutes]
            # rku = rolling_kurtosis[i:i+self.lookback_minutes]
            # macd = macd_indie[i:i+self.lookback_minutes]
            # williams = wpr[i:i+self.lookback_minutes]
            # relative = rsi[i:i+self.lookback_minutes]
            # ichi = ichimoku[i:i+self.lookback_minutes]
            # adx_index = adx[i:i+self.lookback_minutes]

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

            # x_i = np.column_stack((o, h, l, c, v, volat, rsk, rku, macd, williams, relative, ichi, adx_index))
            # column_names = ['o', 'h', 'l', 'c', 'v', 'volat', 'rsk', 'rku', 'macd', 'williams', 'rsi', 'ichi', 'adx']
            x_i = np.column_stack((o, h, l, c, v))
            column_names = ['o', 'h', 'l', 'c', 'v']
            x_i = x_i.flatten()

            for j in range(i+self.lookback_minutes, i+self.lookback_minutes+self.lookforward_minutes
                            ):
                label['forward%s' % str(j)] = (closep[j] - closep[i+self.lookback_minutes -1])/closep[i+self.lookback_minutes -1] * 100

                if j == i+ self.lookback_minutes:
                    down_tot = (label['forward%s' % str(j)] > - down_factor)
                    up_tot = (label['forward%s' % str(j)] > up_factor)    
                else:  
                    d = (label['forward%s' % str(j)] > - down_factor)
                    u = (label['forward%s' % str(j)] > up_factor)
                
                    down_tot = down_tot & d 
                    up_tot = up_tot | u
            
            label['UpDown'] = down_tot & up_tot
            y_i = int(label['UpDown'])

            closeptest_i = closep[i+self.lookback_minutes+self.lookforward_minutes
                            ]
            # dateminute_i = date_minute[i+self.lookback_minutes+self.lookforward_minutes
            #               ]

            X.append(x_i)
            Y.append(y_i)
            close_test.append(closeptest_i)
            # dateminute.append(dateminute_i)

        X, Y = np.array(X), np.array(Y)
        #X_train, X_test, Y_train, Y_test = self.create_Xt_Yt(X, Y, self.TEST_SET)
        feature_names = self.get_feature_names(column_names, self.lookback_minutes)

        return X, Y, feature_names