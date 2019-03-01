'''.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:
@author CL
@email lichendonger@gmail.com
@copyright CL all rights reserved
@created Thu Dec 23 2018 10:12 GMT-0800 (PST)
@last-modified Tue Feb 26 2019 18:02 GMT-0800 (PST)
.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:'''

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
from settings import Config as settings
import time
import datetime
import pandas as pd
from pymongo.uri_parser import parse_uri
from pymongo import MongoClient

class GetDepth:
    '''
    load depth to pandas from mongo
    use function load_depth for best bid and ask
    '''

    def __init__(self):

        self.table_convention = '.depth.level150'
        self.col = ['bid_price', 'bid_size', 'ask_price', 'ask_size', 't']
        self.HUOBI_MONGO_MARKET_URL = settings.HUOBI_MONGO_MARKET_URL
        self.BINANCE_MONGO_MARKET_URL = settings.BINANCE_MONGO_MARKET_URL

    def mongo_query(self, client_db, clause, batch):
        '''
        mongo_query: query coin depth data from mongo
        Params:
                client_db (obj): coin specific db object
                clause: mongo query 
                batch: batch size for each query
        Return: 
                pd_depth (obj, pandas dataframe): pandas depth data 
        '''
        pd_depth = pd.DataFrame(list(client_db.aggregate([
            {'$match': clause},
            {"$project":{
                "_id": 0, 
                "t": 1,
                "ib": {'$divide':[
                    {'$sum':[
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",0]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",1]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",2]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",3]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",4]}, 1]}]
                    }, 
                    {'$sum':[
                        {"$arrayElemAt":[{"$arrayElemAt": ["$a",0]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",0]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$a",1]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",1]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$a",2]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",2]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$a",3]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",3]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$a",4]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",4]}, 1]}]
                    }]
                }, 
                "abp_spread": {'$subtract': [
                    {"$arrayElemAt":[{"$arrayElemAt": ["$a",0]}, 0]}, 
                    {"$arrayElemAt":[{"$arrayElemAt": ["$b",0]}, 0]}
                ]}, 
                "abs_spread": {'$subtract': [
                    {"$arrayElemAt":[{"$arrayElemAt": ["$a",0]}, 1]}, 
                    {"$arrayElemAt":[{"$arrayElemAt": ["$b",0]}, 1]}
                ]}, 
                "midp": {'$avg': [
                    {"$arrayElemAt":[{"$arrayElemAt": ["$a",0]}, 0]}, 
                    {"$arrayElemAt":[{"$arrayElemAt": ["$b",0]}, 0]}
                ]}, 
                "bp_avg": {'$avg': [
                    {"$arrayElemAt":[{"$arrayElemAt": ["$b",0]}, 0]}, 
                    {"$arrayElemAt":[{"$arrayElemAt": ["$b",1]}, 0]}, 
                    {"$arrayElemAt":[{"$arrayElemAt": ["$b",2]}, 0]},
                    {"$arrayElemAt":[{"$arrayElemAt": ["$b",3]}, 0]},  
                    {"$arrayElemAt":[{"$arrayElemAt": ["$b",4]}, 0]}
                ]}, 
                "bs_sum": {'$sum': [
                    {"$arrayElemAt":[{"$arrayElemAt": ["$b",0]}, 1]},
                    {"$arrayElemAt":[{"$arrayElemAt": ["$b",1]}, 1]},
                    {"$arrayElemAt":[{"$arrayElemAt": ["$b",2]}, 1]},
                    {"$arrayElemAt":[{"$arrayElemAt": ["$b",3]}, 1]},
                    {"$arrayElemAt":[{"$arrayElemAt": ["$b",4]}, 1]}
                ]}, 
                "ap_avg": {'$avg': [
                    {"$arrayElemAt":[{"$arrayElemAt": ["$a",0]}, 0]},
                    {"$arrayElemAt":[{"$arrayElemAt": ["$a",1]}, 0]},
                    {"$arrayElemAt":[{"$arrayElemAt": ["$a",2]}, 0]},
                    {"$arrayElemAt":[{"$arrayElemAt": ["$a",3]}, 0]},
                    {"$arrayElemAt":[{"$arrayElemAt": ["$a",4]}, 0]}
                ]}, 
                "as_sum": {'$sum':[
                    {"$arrayElemAt":[{"$arrayElemAt": ["$a",0]}, 1]},
                    {"$arrayElemAt":[{"$arrayElemAt": ["$a",1]}, 1]},
                    {"$arrayElemAt":[{"$arrayElemAt": ["$a",2]}, 1]},
                    {"$arrayElemAt":[{"$arrayElemAt": ["$a",3]}, 1]},
                    {"$arrayElemAt":[{"$arrayElemAt": ["$a",4]}, 1]}
                ]},
                "abp_cumdiff": {'$sum':[
                    {'$subtract':[
                        {"$arrayElemAt":[{"$arrayElemAt": ["$a",0]}, 0]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",0]}, 0]}]
                    },
                    {'$subtract':[
                        {"$arrayElemAt":[{"$arrayElemAt": ["$a",1]}, 0]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",1]}, 0]}]
                    },
                    {'$subtract':[
                        {"$arrayElemAt":[{"$arrayElemAt": ["$a",2]}, 0]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",2]}, 0]}]
                    },
                    {'$subtract':[
                        {"$arrayElemAt":[{"$arrayElemAt": ["$a",3]}, 0]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",3]}, 0]}]
                    },
                    {'$subtract':[
                        {"$arrayElemAt":[{"$arrayElemAt": ["$a",4]}, 0]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",4]}, 0]}]
                    }
                ]},
                "abs_cumdiff": {'$sum':[
                    {'$subtract':[
                        {"$arrayElemAt":[{"$arrayElemAt": ["$a",0]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",0]}, 1]}]
                    },
                    {'$subtract':[
                        {"$arrayElemAt":[{"$arrayElemAt": ["$a",1]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",1]}, 1]}]
                    },
                    {'$subtract':[
                        {"$arrayElemAt":[{"$arrayElemAt": ["$a",2]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",2]}, 1]}]
                    },
                    {'$subtract':[
                        {"$arrayElemAt":[{"$arrayElemAt": ["$a",3]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",3]}, 1]}]
                    },
                    {'$subtract':[
                        {"$arrayElemAt":[{"$arrayElemAt": ["$a",4]}, 1]},
                        {"$arrayElemAt":[{"$arrayElemAt": ["$b",4]}, 1]}]
                    }
                ]}
                }
            },
            {"$sort":{
                "t": 1
                }
            }
            ], allowDiskUse=True).batch_size(batch)))

        return (pd_depth)

    def load_depth(self, exchange, coin, base_currency, 
                    start, end, batch):
        '''
        load_depth: load depth data from Mongo
        Params:
                exchange: 'huobi' or 'binnance'
                coin : string i.e. 'ltcbtc'
                start: start time '28 December 2018'
                end: end time  '29 December 2018'
                batch: int  i.e. 50000
        Return:
                pd_depth (obj, pandas dataframe): pandas depth data
        '''
        time_start = datetime.datetime.now()
        coin = coin + base_currency

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

        depth = db[str(coin) + self.table_convention]

        end = datetime.datetime.strptime(
            end, '%d %B %Y %H:%M').replace(tzinfo=pytz.UTC)
        start = datetime.datetime.strptime(
            start, '%d %B %Y %H:%M').replace(tzinfo=pytz.UTC)

        clause = {'t': {'$gte': start, '$lte': end}}

        # Pandas Data
        pd_depth = self.mongo_query(depth, clause, batch)

        pd_depth.index = pd.to_datetime(
            pd_depth['t'].dt.strftime('%Y-%m-%d %H:%M:%S')).dt.ceil('30S').rename('event_time')
        pd_depth = pd_depth.groupby(pd_depth.index).last()

        pd_depth.drop(columns = ['t'], inplace = True)
        return pd_depth
