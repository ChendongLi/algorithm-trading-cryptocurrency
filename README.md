# Crypocurrency Trend Forecast
This is research/test code (not production) used for predicting price spike and help identify market anomalies.

The project starts from loading kline and limit order book (depth) data. 
Then create selected features to predict spike
Feedforward neunal network and recurrent LSTM model are available to test. 

NOTE: .env is not provided in gitbub. Confidential database access is included in .env. If needed, can be provied by email.

## Install requirements (Python 3.6)
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

![Alt text](/saved_figures/price-volumn--20190307-2147.png?raw=true "Price Volume")

![Alt text](/saved_figures/model-evaluation--20190228-2010.png?raw=true "Model Evaluation")
