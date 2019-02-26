
HUOBI = 'huobi'

BINANCE = 'binance'

BITMEX = 'bitmex'

EX_TRANS_FEE = {
    HUOBI : 0.002,
    BINANCE : 0.00075
}

TRAINING_DATA_BATCH_SIZE = 50000

#original capital to calculate return percentage
ORIG_CAPITAL = 0.8

# annual risk free rate and total trading days for sharp ratio
RISK_FREE_RATE = 0.02
TOTAL_TRADING_DAYS = 365

KLINE_SLIPPAGE_COST = 0.000


# historical days for train optimization
HISTORY_DAY = 5

#valid test score threshold
THRESHOLD = -0.01

#Day Interval for train optimiztion cross optimization
TRAIN_INTERVAL = 0.4 * HISTORY_DAY
VALID_INTERVAL = 0.2 * HISTORY_DAY