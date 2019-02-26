import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten
from keras.models import Sequential, load_model
from keras import backend as K

K.clear_session()

class nn_model:
    def __init__(self, X_train, y_train, X_val, y_val):    
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = Sequential()

    # def f1(self, y_true, y_pred):
    #     y_pred = K.round(y_pred)
    #     tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    #     tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    #     fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    #     fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    #     p = tp / (tp + fp + K.epsilon())
    #     r = tp / (tp + fn + K.epsilon())

    #     f1 = 2*p*r / (p+r+K.epsilon())
    #     f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    #     r = tf.where(tf.is_nan(r), tf.zeros_like(r), r)
    #     return K.mean(r) #rK.mean(f1)

    def f1(self, y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return recall #2*((precision*recall)/(precision+recall+K.epsilon()))

    def f1_loss(self, y_true, y_pred):
        
        tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
        tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2*p*r / (p+r+K.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        r = tf.where(tf.is_nan(r), tf.zeros_like(r), r)
        return  1 - K.mean(f1)

    def fit(self):
        self.model.add(Dense(100, activation= 'relu', 
        input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(200, activation= 'relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(100, activation= 'relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary()
        self.model.compile(optimizer='adam',
            loss= self.f1_loss,
            metrics=[self.f1])

        return self.model.fit(self.X_train,
            self.y_train,
            epochs = 1,
            batch_size=64,
            validation_data = (self.X_val, self.y_val), 
            verbose=1)

    def predict(self, X_test):
        self.model.add(Dense(20, activation= 'relu', 
        input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(40, activation= 'relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(20, activation= 'relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary()
        self.model.compile(optimizer='adam',
            loss= self.f1_loss,
            metrics=[self.f1])
        self.model.fit(self.X_train,
                    self.y_train,
                    epochs = 1,
                    batch_size=64,
                    validation_data = (self.X_val, self.y_val), 
                    verbose=1)

        return self.model.predict(X_test)




class PlotAccuracy:
    def plot(self, history, n_epoch):
        history_dict = history.history
        acc_values = history_dict['acc']
        val_acc_values = history_dict['val_acc']
        epochs = range(1, n_epoch+1)
        plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.show()
