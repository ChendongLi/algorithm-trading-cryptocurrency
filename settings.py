import logging, os
from dotenv import load_dotenv
import json 

# Get the path to the directory this file is in
BASEDIR = os.path.abspath(os.path.dirname(__file__))
PATH = os.path.join(BASEDIR, '.env')

# load environmental variable from .env file, if file exists
if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    load_dotenv(PATH)
    logging.debug(f'load environmental varialbe from {PATH}')
else:
    logging.debug(
        f'skip loading environmental variable from {PATH}, since it not exists')

class Config(object):
    # Connect the path with your '.env' file name
    HUOBI_MONGO_MARKET_URL = os.getenv('HUOBI_MONGO_MARKET_URL')
    BINANCE_MONGO_MARKET_URL = os.getenv('BINANCE_MONGO_MARKET_URL')
