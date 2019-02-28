'''.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:
@author CL
@email lichendonger@gmail.com
@copyright CL all rights reserved
@created Thu Dec 23 2018 15:10 GMT-0800 (PST)
@last-modified Tue Feb 26 2019 18:20 GMT-0800 (PST)
.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:'''

import logging
import utils.logsetup
from tabulate import tabulate
import datetime
import pandas as pd 
import numpy as np 
import json
import os

from dao.load_data import DoubleStrategyLoadData as train
from dao.load_depth import GetDepth
from chart.chart import chart

from getXY.get_XY_depth import DataPrepareForXY as create_XY 
from forecast.NN import Model

import logging
import utils.logsetup

logger = logging.getLogger(__name__)

def main(model = 'lstm'):
    '''
    run the whole process
    '''
    configs = json.load(open('config.json', 'r'))

    ####################
    ## Load Data
    pd_kline = train().coin_kline(
        coin = configs['data']['coin'][0], 
        base_currency = configs['data']['coin'][1], 
        start = configs['data']['date'][0], 
        end = configs['data']['date'][1], 
        exchange=configs['data']['exchange'],
        batch=configs['data']['data_batch'])
    logger.info(tabulate(pd_kline.head(5), headers = 'keys', tablefmt="psql"))

    pd_depth = GetDepth().load_depth(
        exchange = configs['data']['exchange'], 
        coin = configs['data']['coin'][0], 
        base_currency = configs['data']['coin'][1], 
        start = configs['data']['date'][0], 
        end = configs['data']['date'][1],
        batch=configs['data']['data_batch']
    )
    logger.info(tabulate(pd_depth.head(5), headers = 'keys', tablefmt="psql"))

    pd_kd = pd.concat([
        pd_kline, 
        pd_depth],
        axis = 1, 
        join = 'inner')
    logger.info(tabulate(pd_kd.head(), headers = 'keys', tablefmt="psql"))

    ####################
    ## get X and Y
    X, Y, feature_names, price, date_minute =  create_XY(
                    train_val = configs['data']['train_val_split'], 
                    train_test= configs['data']['train_test_split']
                    ).get_XY(
                            data_original = pd_kd, 
                            lookback_minutes= configs['data']['lookback_minutes'], 
                            lookforward_minutes= configs['data']['lookforward_minutes'], 
                            up_factor = configs['data']['up_factor'], 
                            down_factor= configs['data']['down_factor'],
                            step = configs['data']['step'], 
                            model = model
                            )

    X_train, X_val, X_test, Y_train, Y_val, Y_test = create_XY(
                    train_val = configs['data']['train_val_split'], 
                    train_test= configs['data']['train_test_split']
                    ).train_test_split(
                                        X,
                                        Y
                                        )

    logger.info('positive label train percentage %0.4f' % (len(Y_train[Y_train == 1])/len(Y_train)))
    logger.info('positive label validation percentage %0.4f' % (len(Y_val[Y_val == 1])/len(Y_val)))

    if not os.path.exists(configs[model]['save_dir']): 
        os.makedirs(configs[model]['save_dir'])

    save_fname = os.path.join(configs[model]['save_dir'], 
                '%s-%s.h5' % (datetime.datetime.now().strftime('%d%m%Y-%H%M'), model))

    ####################
    ## train and prediction
    if model == 'nn':
        history = Model(X_train, Y_train, X_val, Y_val).nn_fit(save_fname)

    if model == 'lstm':
        history = Model(X_train, Y_train, X_val, Y_val).lstm_fit(save_fname)

    Y_predict = Model(X_train, Y_train, X_val, Y_val).predict(
            X_val, 
            save_fname)

    return (Y_predict)

if __name__ == '__main__':
    Y_predict = main(model = 'lstm')


