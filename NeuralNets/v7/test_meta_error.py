# -*- coding: utf-8 -*-

#%%

from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM,\
                 TimeDistributed, Dense, Bidirectional,\
                 Lambda, RepeatVector, Layer, Concatenate
from keras.layers import concatenate as Concat
from keras.metrics import categorical_accuracy, mean_absolute_error
from keras.losses import mean_squared_error, categorical_crossentropy, mean_absolute_error
import keras.backend as K
from keras.utils import to_categorical

import numpy as np
import numpy.random as rand

import pickle
from collections import Counter



#%%

# differentiable function that learns to
# convert one-hot vectors to indices
class PredictInds(Model):
    def __init__(self, m, V, compile_now=True):
        
        one_hot_inputs = Input(shape=(m, V))

        ind_outputs = TimeDistributed(Dense(1))(one_hot_inputs)
        
        super().__init__(inputs=one_hot_inputs, outputs=ind_outputs)
        
        if compile_now:
            self.compile_default()
            
            
    def compile_default(self):
        self.compile(optimizer="adam", 
                     loss=mean_absolute_error,
                     metrics=[mean_absolute_error])      
        
        
    def __call__(self, inputs, mask=None):
        print("PREDINDS", K.shape(inputs))
        super().__call__(inputs, mask=mask)



class TestModel(Model):
    def __init__(self, ind_predictor, m, V, embed_size, lstm_size, compile_now=True):
        
        embedding_prime = TimeDistributed(Dense(embed_size))
        
        input_seq = Input(shape=(m, V))
        
        embedded_prime = embedding_prime(input_seq)
        
        processed = LSTM(lstm_size, return_sequences=True)(embedded_prime)        
        
        preds = TimeDistributed(Dense(V, activation="softmax"))(processed)
        
        print("SHAPES", 
              (m, V),
              K.shape(input_seq),
              K.shape(embedded_prime),
              K.shape(processed),
              K.shape(preds))
        
        print("--"*30)
        
        
        super().__init__(inputs=input_seq, outputs=preds)
        
        
        self.ind_predictor = ind_predictor
        self.embedding_prime = embedding_prime
        
        
        if compile_now:
            self.compile_default()
            
            
    def compile_default(self):
        self.compile(optimizer="adam", 
                     loss=self.diff_loss,
                     metrics=[categorical_accuracy])

    def diff_loss(self, y_true, y_pred):
        print(K.shape(y_true))
        print(K.shape(y_pred))
        true_inds = self.ind_predictor(y_true)
        pred_inds = self.ind_predictor(y_pred)
        return K.mean(true_inds - pred_inds)
    
    

#    def custom_loss(self, y_true, y_pred):
#        return K.cast(self.mean_diff(y_true, y_pred), dtype="float")
#        #return categorical_crossentropy(y_true, y_pred) #+\
#                    #K.cast(self.mean_diff(y_true, y_pred), dtype="float")    
        
#    def embedding_diff(self, y_true, y_pred):
#        true_inds = self.ind_predictor(y_true)
#        pred_inds = self.ind_predictor(y_pred)
#        
#        true_emb = 
#        
#        return mean_squared_error(true_inds)
#        
#        
#
#    def mean_diff(self, y_true, y_pred):
#        true_inds = K.argmax(y_true, axis=-1)
#        pred_inds = K.argmax(y_pred, axis=-1)
#        
#        true_mean = K.mean(true_inds, axis=-1)
#        pred_mean = K.mean(pred_inds, axis=-1)
#        
#        return K.abs(true_mean - pred_mean)
#        
        
        
        
#%%
       
V = 4  
sl = 5
        
test_data = rand.randint(V, size=(10, 5))
test_data_cat = to_categorical(test_data, num_classes=V)


means = np.mean(test_data, axis=-1)
#%%
        
m = TestModel(ind_pred, sl, V, 2, 6)

#%%


m.fit(x=test_data, y=test_data_cat, epochs=10, verbose=2)



#%%

r = rand.randint(V, size=(1, 5))

test_preds = m.predict(test_data)


test_pred_inds = np.argmax(test_preds, axis=-1)

print(test_pred_inds, test_data)
print()
print(np.mean(test_pred_inds, axis=-1), np.mean(test_data, axis=-1))






#%% TRAIN PREDINDS


ind_pred = PredictInds(sl, V)


#%%

test_data_reshaped = test_data.reshape((*test_data.shape, 1))


#%%
ind_pred.fit(x=test_data_cat, y=test_data_reshaped, epochs=2, verbose=2)



#%%

r = rand.randint(V, size=(1, 5))
r_cat = to_categorical(r, num_classes=V)

test_preds = ind_pred.predict(r_cat)

print(test_preds.reshape((1, 5)), r)



