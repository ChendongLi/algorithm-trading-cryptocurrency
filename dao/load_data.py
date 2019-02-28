'''.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:
@author CL
@email lichendonger@gmail.com
@copyright CL all rights reserved
@created Thu Dec 22 2018 12:16:25 GMT-0800 (PST)
@last-modified Tue Feb 26 2019 17:05:19 GMT-0800 (PST)
.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:'''

import logging
import utils.logsetup
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

logger = logging.getLogger(__name__)

class DoubleStrategyLoadData:

    def __init__(self):

        self.HUOBI_MONGO_MARKET_URL = 'mongodb://admin:admin@trade.questflex.com:32017/huobi-market-data'
        self.BINANCE_MONGO_MARKET_URL = 'mongodb://admin:admin@trade.questflex.com:32017/binance-market-data'

    def load_mongo(self,
                   coin,
                   clause,
                   exchange,
                   batch= TRAINING_DATA_BATCH_SIZE):
        '''
        load_mongo: pull data from mongo data
        Params: 
                coin (str): name of coin i.e. 'ltc'
                clause (str): clause to enter start, end time
                exchange (str): name of exchange i.e. BINANCE
                batch (int): batch size to pull data from Mongo DB
        Return: 
                kline (obj, pandas dataframe):  kline raw data from Mongo DB
        '''

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
            kline = pd.DataFrame(data)
            kline.rename(columns={
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
            return (kline)

    def get_vwap(self, pd_coin):
        '''
        get_vwap: get volume weighted average price
        Params: 
                pd_coin (obj, pandas dataframe): load coin data from load_mongo
        Return: 
                pd_coin_vwap (obj, pandas dataframe):  volume weighted average price
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
        coin_kline: load single coin and return pandas data
        Params: 
                coin (str): coin name  i.e. 'ltc'
                base_currency (str): coin base currency i.e. 'usdt'
                start (str): data start time string i.e. '1 November 2018 00:00'
                end (str): data end time string i.e. '18 February 2019 00:00'
                exchange (str): exchange name i.e. 'BINANCE'
        Return: 
                coin_kline (obj, pandas dataframe): pandas dataframe for volume 
                weighted average price
        '''

        coin = coin + base_currency
        TR_FEE = float(EX_TRANS_FEE[exchange])

        end = datetime.datetime.strptime(
            end, '%d %B %Y %H:%M').replace(tzinfo=pytz.UTC)
        start = datetime.datetime.strptime(
            start, '%d %B %Y %H:%M').replace(tzinfo=pytz.UTC)

        time_clause = {'s': {'$lt': end, '$gte': start}}
        logger.info(f'Time start:{start}, end:{end}')
        logger.info(f'Load kline coin:{coin}')

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
        getVol: get first return log difference Volatility 
        Params: 
                data (obj, pandas dataframe): price data
                span0 (int): moving average window length
                col (str): the name of the colume to get volatility 
        Return: 
                vol (array): volatility array for the selected column 
        '''
        close = data[col]
        df0=close.index.searchsorted(close.index-pd.Timedelta(minutes=1))
        df0=df0[df0>0]   
        df0=pd.Series(close.index[df0],  
                    index=close.index[close.shape[0]-df0.shape[0]:])
        try:
            diff= (np.log(close.loc[df0.index].values) 
                    - np.log(close.loc[df0.values].values)) # daily rets
            
            diff = pd.Series(diff, index = df0.index[(df0.shape[0] - len(diff)):])[2:]

        except Exception as e:
            logger.info(f'error: {e}\nplease confirm no duplicate indices')

        vol=diff.ewm(span=span0).std()[span0:].dropna().rename('diffVol')

        return vol

    def washData(self, data, span0 = 1400, col = 'diff'):
        '''
        washData: 
                filter out events move no more than the mean of minute volatility
        Params: 
                data (obj, pandas dataframe): price data
                span0 (int): moving average window length
                col (str): the name of the colume to get volatility 
        Return: 
                clean_data (obj, pandas dataframe): price data after filtering                
        '''
        # get volatility 
        h = self.getVol(data = data, span0 = span0, col = col).mean()

        Events, sPos, sNeg = [], 0, 0
        diff = np.log(data[col]).diff()

        for i in diff.index[1:]:
            try:
                pos, neg = float(sPos+diff.loc[i]), float(sNeg+diff.loc[i])
            except Exception as e:
                logger.info(e)
                logger.info(sPos+diff.loc[i], type(sPos+diff.loc[i]))
                logger.info(sNeg+diff.loc[i], type(sNeg+diff.loc[i]))
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
        '''
        load_pair: join coin pair data for both kline and depth
        Params:
                coin_1 (str): coin name i.e. 'ltc' 
                coin_2 (str): coin name i.e. 'eth'
                base_currency (str): coin name i.e. 'usdt'
                start: data start time string i.e. '1 November 2018 00:00'
                end: data end time string i.e. '18 February 2019 00:00'
                exchange (str): exchange name i.e. 'BINANCE'
        Return:
                data (obj, pandas dataframe): price data for two coins (pair)
        '''
        coin1 = coin_1 + base_currency
        coin2 = coin_2 + base_currency
        # history_hours = 24 * int(settings.DOUBLE_TRAIN_PARAMS['history_day'])
        TR_FEE = float(EX_TRANS_FEE[exchange])

        end = datetime.datetime.strptime(
            end, '%d %B %Y').replace(tzinfo=pytz.UTC)
        start = datetime.datetime.strptime(
            start, '%d %B %Y').replace(tzinfo=pytz.UTC)

        time_clause = {'s': {'$lt': end, '$gte': start}}
        logger.info(f'Time start:{start}, end:{end}')

        logger.info(f'Load kline coin1:{coin1}')
        coin1_kline = self.load_mongo(
            coin=coin1,
            clause=time_clause,
            exchange=exchange)

        logger.info(f'Load kline coin2:{coin2}')
        coin2_kline = self.load_mongo(
            coin=coin2,
            clause=time_clause,
            exchange=exchange)

        # get kline data by each minute
        pd_coin1_kline = self.get_vwap(
            coin1_kline).add_suffix('_coin1') 
        pd_coin2_kline = self.get_vwap(
            coin2_kline).add_suffix('_coin2')

        # get depth data by each minute
        logger.info(f'Load depth coin1:{coin1}')
        pd_coin1_depth = GetDepth().load_depth(
            exchange=exchange,
            coin=coin_1,
            base_currency=base_currency, 
            start=start,
            end=end
        ).add_suffix('_coin1')

        logger.info(f'Load depth coin2:{coin2}')
        pd_coin2_depth = GetDepth().load_depth(
            exchange=exchange,
            coin=coin_2,
            base_currency=base_currency, 
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