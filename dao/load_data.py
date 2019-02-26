from pymongo import MongoClient
from pymongo.uri_parser import parse_uri
import pandas as pd
import datetime
import pytz
from pytz import timezone
import numpy as np
import math as math
from tabulate import tabulate
from dao.load_depth import GetDepth
from dao.constant import EX_TRANS_FEE, HUOBI, BINANCE, BITMEX, TRAINING_DATA_BATCH_SIZE
from indicator.indicator import directional_movement_index, average_true_range
from pyti import directional_indicators
from pyti import bollinger_bands
# from indicator import keltner
# from indicator.stationary import stationary 

class DoubleStrategyLoadData:

    def __init__(self):

        self.HUOBI_MONGO_MARKET_URL = 'mongodb://admin:admin@trade.questflex.com:32017/huobi-market-data'
        self.BINANCE_MONGO_MARKET_URL = 'mongodb://admin:admin@trade.questflex.com:32017/binance-market-data'

    def load_mongo(self,
                   coin,
                   clause,
                   exchange,
                   batch= TRAINING_DATA_BATCH_SIZE):

        if exchange == HUOBI:
            mongo_market_url = self.HUOBI_MONGO_MARKET_URL
        elif exchange == BINANCE:
            mongo_market_url = self.BINANCE_MONGO_MARKET_URL

        if not mongo_market_url:
            raise Exception("Empty exchange mongo url")

        default_db_name = f'{exchange}-market-data'

        parsed = parse_uri(mongo_market_url)

        dbname = default_db_name if parsed['database'] is None else parsed[
            'database']
        client = MongoClient(mongo_market_url, tz_aware=True)

        # Get the MongoDB database
        db = client[dbname]

        kline = db[str(coin) + '.kline.1min']

        data = list(
            kline.aggregate([{
                '$match': clause
            }, 
            # {
            #     '$sort': {
            #         'cnt': -1
            #     }
            # }, {
            #     '$group': {
            #         '_id': '$s',
            #         'record': {
            #             '$first': '$$ROOT'
            #         }
            #     }
            # }, {
            #     '$replaceRoot': {
            #         'newRoot': '$record'
            #     }
            # }, 
            {
                '$sort': {
                    't': 1
                }
            }], allowDiskUse=True).batch_size(batch))

        if not data:
            raise Exception(
                f'Loaded empty data from Mongo database: {mongo_market_url} between start: {clause["t"]["$gte"].isoformat()} and end: {clause["t"]["$lt"].isoformat()}'
            )
        else:
            temp = pd.DataFrame(data)
            temp.rename(columns={
                'o': 'open', 
                'c': 'close', 
                'h': 'high', 
                'l': 'low', 
                'v': 'vol', 
                'a': 'amount', 
                'cnt': 'count',
                's': 'start', 
                'e': 'end', 
                't': 'event_time'
            }, inplace=True)
            return (temp)

    def get_vwap(self, pd_coin):
        '''
        get volume weighted average price
        '''
        # Get Volume Weighte Average Price
        pd_coin['vol'] = pd.to_numeric(pd_coin['vol'])
        pd_coin['amount'] = pd.to_numeric(pd_coin['amount'])

        pd_coin.index = pd.to_datetime(pd_coin['start'])
        pd_coin = pd_coin.groupby(pd_coin.index).last()
        pd_coin = pd_coin[pd_coin['amount'] != 0].dropna()
        pd_coin['vwap'] = pd_coin['vol'] / pd_coin['amount']

        pd_coin_vwap = pd.DataFrame({
            'vwap': pd_coin['vwap'], 
            'high': pd_coin['high'], 
            'low': pd_coin['low'], 
            'open': pd_coin['open'], 
            'close': pd_coin['close']}, index=pd_coin['start'])

        return (pd_coin_vwap)

    def coin_kline(self, coin, base_currency, start, end, exchange):
        '''
        load single coin and return pandas data
        '''

        coin = coin + base_currency
        TR_FEE = float(EX_TRANS_FEE[exchange])

        end = datetime.datetime.strptime(
            end, '%d %B %Y %H:%M').replace(tzinfo=pytz.UTC)
        start = datetime.datetime.strptime(
            start, '%d %B %Y %H:%M').replace(tzinfo=pytz.UTC)

        time_clause = {'s': {'$lt': end, '$gte': start}}
        print(f'Time start:{start}, end:{end}')
        print(f'Load kline coin:{coin}')
        coin_kline = self.load_mongo(
            coin=coin,
            clause=time_clause,
            exchange=exchange)
        
        coin_kline.index = pd.to_datetime(coin_kline['event_time'].dt.strftime('%Y-%m-%d %H:%M:%S')).dt.ceil('30S')
        coin_kline = coin_kline.groupby(coin_kline.index).last()

        coin_kline.drop([
            '_id', 'count', 'start', 'end', 'event_time'
        ], 
        axis = 1, inplace = True)
        coin_kline.dropna(inplace = True)

        return coin_kline

    def getVol(self, data, span0=100, col = 'diff'):
        '''
        get first return log difference Volatility 
        '''
        close = data[col]
        df0=close.index.searchsorted(close.index-pd.Timedelta(minutes=1))
        df0=df0[df0>0]   
        df0=pd.Series(close.index[df0],  
                    index=close.index[close.shape[0]-df0.shape[0]:])
        try:
            #diff=close.loc[df0.index]/close.loc[df0.values].values-1 # daily rets
            diff= (np.log(close.loc[df0.index].values) 
            - np.log(close.loc[df0.values].values)) # daily rets
            
            diff = pd.Series(diff, index = df0.index[(df0.shape[0] - len(diff)):])[2:]

        except Exception as e:
            print(f'error: {e}\nplease confirm no duplicate indices')

        vol=diff.ewm(span=span0).std()[span0:].dropna().rename('diffVol')

        return vol

    def washData(self, data, span0 = 1400, col = 'diff'):
        '''
        filter out events move no more than the mean of minute volatility
        '''
        # get volatility 
        h = self.getVol(data = data, span0 = span0, col = col).mean()

        Events, sPos, sNeg = [], 0, 0
        diff = np.log(data[col]).diff()

        for i in diff.index[1:]:
            try:
                pos, neg = float(sPos+diff.loc[i]), float(sNeg+diff.loc[i])
            except Exception as e:
                print(e)
                print(sPos+diff.loc[i], type(sPos+diff.loc[i]))
                print(sNeg+diff.loc[i], type(sNeg+diff.loc[i]))
                break
            sPos, sNeg=max(0., pos), min(0., neg)
            
            if sNeg<-h:
                sNeg=0;Events.append(i)
            elif sPos>h:
                sPos=0;Events.append(i)
        
        tEvents = pd.DatetimeIndex(Events)
        clean_data=data.loc[tEvents]

        return clean_data

    def load_pair(self, coin_1, coin_2, base_currency, 
                    start, end, exchange):

        coin1 = coin_1 + base_currency
        coin2 = coin_2 + base_currency
        # history_hours = 24 * int(settings.DOUBLE_TRAIN_PARAMS['history_day'])
        TR_FEE = float(EX_TRANS_FEE[exchange])

        end = datetime.datetime.strptime(
            end, '%d %B %Y').replace(tzinfo=pytz.UTC)
        start = datetime.datetime.strptime(
            start, '%d %B %Y').replace(tzinfo=pytz.UTC)

        time_clause = {'s': {'$lt': end, '$gte': start}}
        print(f'Time start:{start}, end:{end}')
        print(f'Load kline coin1:{coin1}')
        coin1_kline = self.load_mongo(
            coin=coin1,
            clause=time_clause,
            exchange=exchange)
        print(f'Load kline coin2:{coin2}')
        coin2_kline = self.load_mongo(
            coin=coin2,
            clause=time_clause,
            exchange=exchange)

        # get kline data by each minute
        pd_coin1_kline = self.get_vwap(
            coin1_kline).add_suffix('_coin1') #rename(columns={'vwap': 'coin1'})
        pd_coin2_kline = self.get_vwap(
            coin2_kline).add_suffix('_coin2') #rename(columns={'vwap': 'coin2'})

        # get depth data by each minute
        print(f'Load depth coin1:{coin1}')
        pd_coin1_depth = GetDepth().load_depth(
            exchange=exchange,
            coin=coin1,
            start=start,
            end=end
        ).add_suffix('_coin1')

        print(f'Load depth coin2:{coin2}')
        pd_coin2_depth = GetDepth().load_depth(
            exchange=exchange,
            coin=coin2,
            start=start,
            end=end
        ).add_suffix('_coin2')

        # Make sure all coin same length, therefore inner concat coin1 and coin2
        data = pd.concat([
            pd_coin1_kline,
            pd_coin2_kline,
            pd_coin1_depth,
            pd_coin2_depth],
            axis=1,
            join='inner')

        return(data)

    def train_model(self, data, windowHours):
        time_start = datetime.datetime.now()
        windowString = str(windowHours) + 'H'

        pd_data = pd.DataFrame(
            {
                'coin1': data['vwap_coin1'],
                'coin2': data['vwap_coin2'],
                'bid_price_coin1': data['bid_price_coin1'],
                'ask_price_coin1': data['ask_price_coin1'],
                'bid_price_coin2': data['bid_price_coin2'],
                'ask_price_coin2': data['ask_price_coin2'],
                'diff': data['vwap_coin1'] / data['vwap_coin2'],
                'high': data['high_coin1'] / data['low_coin2'],
                'low': data['low_coin1'] / data['high_coin2'],
                'open': data['open_coin1'] / data['open_coin2'],
                'close': data['close_coin1'] / data['close_coin2'],
                'diff_bid_ask': data['bid_price_coin1'] / data['ask_price_coin2'],
                'diff_ask_bid': data['ask_price_coin1'] / data['bid_price_coin2']
            },
            index=data.index
        )

        pd_data['volatility_value'] = pd_data['diff'].rolling(windowString).std()
        pd_data['volatility_rolling_value'] = pd_data['volatility_value'].rolling(windowString).mean()

        pd_data['mean'] = pd_data['diff'].rolling(windowString).mean()
        pd_data['volatility'] = pd_data['volatility_value']/pd_data['mean']
        pd_data['volatility_rolling'] = pd_data['volatility_value'].rolling(windowString).mean()/pd_data['mean']


        pd_data['adx'] = directional_indicators.average_directional_index(
            list(pd_data['close']),
            list(pd_data['high']), 
            list(pd_data['low']), 
            60
            )
        pd_data['adx_quantile'] = pd_data['adx'].rolling(60*24).quantile(0.45
        , interpolation='lower')

        pd_data['adx_quantile_20%'] = pd_data['adx'].rolling(60*24).quantile(0.2
        , interpolation='lower')

        pd_data = directional_movement_index(pd_data, 60)

        training_start = pd_data.index.min() + datetime.timedelta(hours=windowHours)
        pd_data = pd_data[pd_data.index >= training_start]
        pd_data = pd_data.dropna()

        print(
            f'time usage: {str(datetime.datetime.now() - time_start)} seconds')

        return pd_data

    def train_research(self, data, windowHours):
        time_start = datetime.datetime.now()
        windowString = str(windowHours) + 'H'

        pd_data = pd.DataFrame(
            {
                'coin1': data['vwap_coin1'],
                'coin2': data['vwap_coin2'],
                'bid_price_coin1': data['bid_price_coin1'],
                'ask_price_coin1': data['ask_price_coin1'],
                'bid_price_coin2': data['bid_price_coin2'],
                'ask_price_coin2': data['ask_price_coin2'],
                'diff': data['vwap_coin1'] / data['vwap_coin2'],
                'high': data['high_coin1'] / data['vwap_coin2'],
                'low': data['vwap_coin1'] / data['high_coin2'],
                'open': data['open_coin1'] / data['open_coin2'],
                'close': data['close_coin1'] / data['close_coin2'],
                'diff_bid_ask': data['bid_price_coin1'] / data['ask_price_coin2'],
                'diff_ask_bid': data['ask_price_coin1'] / data['bid_price_coin2']
            },
            index=data.index
        )

        pd_data['volatility_value'] = pd_data['diff'].ewm(span= 60*3).std()
        pd_data['volatility_rolling_value'] = pd_data['volatility_value'].ewm(span= 3*60).mean()

        pd_data['mean'] = pd_data['diff'].rolling(
            6*60).mean()
        pd_data['volatility'] = pd_data['volatility_value']/pd_data['mean']
        pd_data['volatility_rolling'] = pd_data['volatility_value'].rolling(
            3*60).quantile(0.8, interpolation='lower')/pd_data['mean']

        # pd_data['skew'] = pd_data['diff'].rolling(
        #     6*60).skew()

        # pd_data['skew_upper'] = pd_data['skew'].rolling(
        #     6*60).quantile(0.7)

        # pd_data['skew_lower'] = pd_data['skew'].rolling(
        #     6*60).quantile(0.3)

        # pd_data['upper_bb'] = bollinger_bands.upper_bollinger_band(list(pd_data['close']), 6*60, std_mult= 1.5)
        # pd_data['lower_bb'] = bollinger_bands.lower_bollinger_band(list(pd_data['close']), 6*60, std = 1.5)

        # pd_data['upper_kb'] = keltner.upper_band(
        #     list(pd_data['close']),
        #     list(pd_data['high']), 
        #     list(pd_data['low']), 
        #     60*6,
        #     5)

        # pd_data['lower_kb'] = keltner.lower_band(            
        #     list(pd_data['close']),
        #     list(pd_data['high']), 
        #     list(pd_data['low']), 
        #     60*6,
        #     5)

        # pd_data['stationary'] = pd_data['diff'].rolling(40*24).apply(lambda x: stationary(x)[0], raw = False)
        pd_data['adx'] = directional_indicators.average_directional_index(
            list(pd_data['close']),
            list(pd_data['high']), 
            list(pd_data['low']), 
            90
            )
        pd_data['adx_quantile'] = 5.0
        #pd_data['adx'].rolling(60*6).mean()
        #quantile(0.45
        #, interpolation='lower')

        pd_data['adx_quantile_20%'] = pd_data['adx'].rolling(60*72).quantile(0.2
        , interpolation='lower')

        #pd_data = directional_movement_index(pd_data, 60)

        training_start = pd_data.index.min() + datetime.timedelta(hours=windowHours)
        pd_data = pd_data[pd_data.index >= training_start]
        pd_data = pd_data.dropna()

        print(
            f'time usage: {str(datetime.datetime.now() - time_start)} seconds')

        return pd_data