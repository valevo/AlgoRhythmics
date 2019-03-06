# -*- coding: utf-8 -*-


#%%
from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM,\
                 TimeDistributed, Dense, Bidirectional,\
                 Lambda, RepeatVector, Layer
from keras.metrics import categorical_accuracy, mean_absolute_error
from keras.losses import mean_squared_error, categorical_crossentropy, mean_absolute_error
import keras.backend as K
from keras.utils import to_categorical

import numpy as np
import numpy.random as rand

import pickle
from collections import Counter


class MeanNet(Model):
    def __init__(self, vlen):
        self.n_calls = 0
        inputs = Input(shape=(vlen, ))
        
        mean_layer = Dense(1)

        mean_pred = mean_layer(inputs)

        super().__init__(inputs=inputs, outputs=mean_pred)
        
        self.mean_layer = mean_layer
        self.mean_pred = mean_pred
        
        
    def my_loss(self, y_true, y_pred):
        print("MEAN LOSS")
        self.n_calls += 1
#        weights_L2 = K.sum(self.mean_layer.weights)
        return K.sum(K.pow(y_true - self.mean_pred, 4))
    
        
class VarNet(Model):
    def __init__(self, vlen):
        inputs = Input(shape=(vlen, ))

        var_pred = Dense(1)(inputs)

        super().__init__(inputs=inputs, outputs=var_pred)
        
        
    def my_loss(self, y_true, y_pred):
        return K.mean(K.abs(y_true-y_pred))
            
    



#%%

l = 5

mn = MeanNet(l)
vn = VarNet(l)


#%%

inputs = Input(shape=(l, ))

m_pred = mn(inputs)
v_pred = vn(inputs)

m = Model(inputs=inputs, outputs=[m_pred, v_pred])


#%%


m.compile("adam", loss=[mn.my_loss, vn.my_loss])


#%%

data_x = rand.randint(1, 5, size=(8, l))

data_y = [np.mean(data_x, axis=-1), np.var(data_x, axis=-1)]

#%%

m.fit(x=data_x, y=data_y, epochs=10, verbose=2)
