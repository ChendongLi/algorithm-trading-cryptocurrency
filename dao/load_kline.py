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
import json
import numpy as np
from settings import Config as settings

logger = logging.getLogger(__name__)

class GetKline:

    def __init__(self):

        self.HUOBI_MONGO_MARKET_URL = settings.HUOBI_MONGO_MARKET_URL
        self.BINANCE_MONGO_MARKET_URL = settings.BINANCE_MONGO_MARKET_URL
        self.configs = json.load(open('config.json', 'r'))

    def load_mongo(self,
                   coin,
                   clause,
                   exchange,
                   batch):
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

        if exchange == 'huobi':
            mongo_market_url = self.HUOBI_MONGO_MARKET_URL
        elif exchange == 'binance':
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

    def coin_kline(self, coin, base_currency, start, end, exchange, batch):
        '''
        coin_kline: load single coin and return pandas data
        Params: 
                coin (str): coin name  i.e. 'ltc'
                base_currency (str): coin base currency i.e. 'usdt'
                start (str): data start time string i.e. '1 November 2018 00:00'
                end (str): data end time string i.e. '18 February 2019 00:00'
                exchange (str): exchange name i.e. 'BINANCE'
                batch (int): batch size to pull data from Mongo DB
        Return: 
                coin_kline (obj, pandas dataframe): pandas dataframe for volume 
                weighted average price
        '''

        coin = coin + base_currency
        TR_FEE = float(self.configs['data']['transaction_fee'][exchange])

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
            exchange=exchange,
            batch=batch)
        
        coin_kline.index = pd.to_datetime(coin_kline['event_time'].dt.strftime('%Y-%m-%d %H:%M:%S')).dt.ceil('30S')
        coin_kline = coin_kline.groupby(coin_kline.index).last()

        coin_kline.drop([
            '_id', 'count', 'start', 'end', 'event_time'
        ], 
        axis = 1, inplace = True)
        coin_kline.dropna(inplace = True)

        return coin_kline