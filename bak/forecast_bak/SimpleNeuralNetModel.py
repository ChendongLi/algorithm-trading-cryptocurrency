#%%
import numpy as np
import pandas as pd
from tabulate import tabulate

from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Reshape, Lambda
from keras.layers import Input, concatenate #, Merge
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Convolution1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, RepeatVector, AveragePooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from keras.constraints import *
from keras import regularizers
from keras import losses
from keras.layers.noise import *
from keras import backend as K 

#%%
class NeuralNetModel:
    """    
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average."""

    def __init__(self):
        self.dir = '/Users/cli/Data/pair_selection/forecast_skewness'
        self.test_set = 0.3

    def nn_model(self, X, Y, X_train, Y_train, X_test, Y_test):

        main_input = Input(shape=(X.shape[1], ), name='main_input')
        x = GaussianNoise(0.05)(main_input)
        #x = main_input
        x = Lambda(lambda x: K.clip(x, min_value=-1, max_value=1))(x)
        x = Dense(64, activation='relu')(x)
        x = GaussianNoise(0.05)(x)
        #x= Flatten()
        output = Dense(1, activation = "linear", name = "out")(x)

        final_model = Model(inputs=[main_input], outputs=[output])

        opt = Adam(lr=0.002)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss'
        , factor=0.9, patience=10, min_lr=0.000001, verbose=1)

        # final_model = Sequential ()
        # input_shape =(X.shape[1], ) 
        # final_model.add (LSTM ( 400,  activation = 'relu', inner_activation = 'hard_sigmoid' , 
        # bias_regularizer=L1L2(l1=0.01, l2=0.01),  input_shape =(X.shape[1], 1), return_sequences = False ))
        # final_model.add(Dropout(0.3))
        # final_model.add (Dense (output_dim =1, activation = 'linear', activity_regularizer=regularizers.l1(0.01)))
        # adam=optimizers.Adam(lr=0.01, beta_1=0.89, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        # final_model.compile (loss ="mean_squared_error" , optimizer = "adam") 

        checkpointer = ModelCheckpoint(monitor='val_loss'
        , filepath= self.dir + '.h5', verbose=1, save_best_only=True)

        final_model.compile(optimizer=opt, 
                    loss='mse')
        final_model.summary()
        
        # for layer in final_model.layers:
        #     print (layer, layer.output_shape)
        #try:


        history = final_model.fit(X, Y, 
                epochs = 40, 
                batch_size = 256, 
                verbose=1, 
                validation_split = 1 - self.test_set, 
                #validation_data=(test_x, test_y),
                callbacks=[reduce_lr, checkpointer],
                shuffle=True)

        final_model.load_weights(self.dir+ '.h5')
        pred = final_model.predict(X_test)

        predicted = pred
        original = Y_test

        return(predicted, original)


