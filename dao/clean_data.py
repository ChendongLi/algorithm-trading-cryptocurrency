'''.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:
@author CL
@email lichendonger@gmail.com
@copyright CL all rights reserved
@created Thu Jan 22 2018 18:16 GMT-0800 (PST)
@last-modified Tue Feb 28 2019 19:06 GMT-0800 (PST)
.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:.:.:.::.:.:'''

import logging
import utils.logsetup
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class CleanKline:
    '''
    clean up data 
    remove noisy data below average volatility
    '''

    def __init__(self, span, col):
        self.span = span 
        self.col = col 

    def getVol(self, data):
        '''
        getVol: get first return log difference Volatility 
        Params: 
                data (obj, pandas dataframe): price data
        Return: 
                vol (array): volatility array for the selected column 
        '''
        close = data[self.col]
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

        vol=diff.ewm(span=self.span).std()[self.span:].dropna().rename('diffVol')

        return vol

    def washData(self, data):
        '''
        washData: 
                filter out events move no more than the mean of minute volatility
        Params: 
                data (obj, pandas dataframe): price data
        Return: 
                clean_data (obj, pandas dataframe): price data after filtering                
        '''
        # get volatility 
        h = self.getVol(data = data).mean()

        Events, sPos, sNeg = [], 0, 0
        diff = np.log(data[self.col]).diff()

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