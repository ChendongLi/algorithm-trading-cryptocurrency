#%%
%load_ext autoreload
%autoreload 2

import numpy as np
from tabulate import tabulate
from dao.load_data import DoubleStrategyLoadData as train
# from chart.chart import chart
from getXY.get_XY import DataPrepareForXY as create_XY 
# from forecast.RandomForestModel import RandomForecastTrendForecasting as rf 
# from forecast.SimpleNeuralNetModel import NeuralNetModel as nn
from forecast.CNN import CNN_Forecast
from dao.constant import EX_TRANS_FEE, HUOBI, BINANCE

#%%
##Load Data
#############################

pd_kline = train().coin_kline(
    coin = 'btc'
    , base_currency = 'usdt'
    , start = '1 December 2018 00:00'
    , end = '1 January 2019 00:00'
    , exchange=BINANCE)

print(tabulate(pd_kline.head(), headers = 'keys', tablefmt="psql"))

#%%
## Prepare Data For XY
#######################################

X, Y, feature_names =  create_XY(
    lookback_minutes=30, lookforward_minutes=10).get_XY(
    data_original = pd_kline, 
    up_factor = 1.0, 
    down_factor= 0.4 
    )

X_train, X_test, y_train, y_test = create_XY(30, 10).train_test_split(X, Y)

print(tabulate(X_train[0:1], headers=feature_names, tablefmt="psql"), tabulate(X_test[0:1], headers=feature_names, tablefmt="psql"), y_train[0:10], y_test[0:10])
print(len(feature_names), len(X_train[0]), len(X_train), len(X_test), len(y_train), len(y_test), np.count_nonzero(y_train), np.count_nonzero(y_test))

#%%
## Run Random Forest
#######################################
# rf().rf_model(
#         X_train = X_train, 
#         X_test = X_test, 
#         y_train = y_train,  
#         y_test = y_test, 
#         n_estimators = 1000, 
#         max_depth = 8, 
#         feature_names = feature_names)

#%%
## Run Neural Net
#######################################
cnn = CNN_Forecast(feature_names)
history = cnn.fit(X_train, y_train, X_test, y_test)

#%%
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.subplot(2,1,1)
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2,1,2)
history_dict = history.history
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#%%
val = train().coin_kline(
    coin = 'btc'
    , base_currency = 'usdt'
    , start = '1 January 2019 00:00'
    , end = '15 January 2019 00:00'
    , exchange=BINANCE)

X_val, Y_val, feature_names =  create_XY(
    lookback_minutes=30, lookforward_minutes=10).get_XY(
    data_original = val, 
    up_factor = 1.0, 
    down_factor= 0.4 
    )

results = cnn.model.evaluate(X_val, Y_val)
print(results)

#%%
from sklearn.metrics import confusion_matrix

print('train predict result')
Y_train_pred = cnn.model.predict(X_train)
print(confusion_matrix(y_train, Y_train_pred))

#%%
#print('validate predict result')
Y_val = cnn.model.predict(X_val)
print(confusion_matrix(Y_val, Y_val))
