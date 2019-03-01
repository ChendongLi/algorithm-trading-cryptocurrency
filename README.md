# Crypocurrency Trend Forecast
This is research/test code (not production) used for predicting trend/spike and help identify market anomalies.

The project starts from loading kline and limit order book (depth) data. 
Then create selected features to predict either price up/down or spike
Feedforward neunal network and recurrent LSTM model are available to test. 

NOTE: .env is not provided in gitbub. Confidential database access is included in .env. If needed, can be provied by email.

## Install requirements
```
pip install -r requirements.txt
```
## Usage: run in the terminal
```
python run.py
```

## Hyperparameter Tunning: run in the terminal
```
python ./forecast/Tune_NN.py
```
## Parameter
Parameters can be adjusted in config.json
