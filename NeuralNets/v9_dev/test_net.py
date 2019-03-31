# -*- coding: utf-8 -*-

#%%
from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM,\
                 TimeDistributed, Dense, Bidirectional,\
                 Lambda, RepeatVector, Layer, Conv1D, Reshape
from keras.layers import concatenate as Concat
from keras.metrics import categorical_accuracy, mean_absolute_error
from keras.losses import mean_squared_error, categorical_crossentropy,\
                    mean_absolute_error
import keras.backend as K
from keras.utils import to_categorical, plot_model

import numpy as np
import numpy.random as rand

import pickle
from collections import Counter

#%%

class AVGPred(Model):
    def __init__(self, V, seq_len, hidden_size):
        seq_one_hot = Input(shape=(seq_len, V))
        
#        emb = Embedding(input_dim=V,
#                        output_dim=embed_size,
#                        input_length=seq_len)(seq)
        
        proc = Bidirectional(LSTM(hidden_size), merge_mode="concat")(seq_one_hot)
        
        out = Dense(1)(proc)
        
        super().__init__(seq_one_hot, out, name="avg_hat")
        
        self.compile("adam",
                     loss=mean_squared_error,
                     metrics=[mean_absolute_error])
        
        
    def freeze(self):
        for l in self.layers:
            l.trainable = False
            
            
            
class SeqModel(Model):
    def __init__(self, seq_len, embed_size, hidden_size, V, avg_net):
        seq = Input(shape=(seq_len, ))
        
        emb = Embedding(input_dim=V,
                        output_dim=embed_size,
                        input_length=seq_len)(seq)
        
        proc = LSTM(hidden_size, return_sequences=True)(emb)
        
        proc2 = LSTM(hidden_size, return_sequences=True)(proc)
        
        out = TimeDistributed(Dense(V), name="out")(proc2)
        
        pred_avg = avg_net(out)
        
        super().__init__(seq, [out, pred_avg])
        
#        self.compile("adam",
#                     loss=[categorical_crossentropy, mean_squared_error],
#                     metrics=[categorical_accuracy, mean_absolute_error])
                
        self.compile("adam",
                     loss={"out": categorical_crossentropy, 
                           "avg_hat": mean_squared_error},
                     metrics={"out": categorical_accuracy, 
                              "avg_hat": mean_absolute_error})
        
        
#%%
        
def dirichlet_noise(one_hot, prep_f=lambda v: v*10+1):
    return rand.dirichlet(prep_f(one_hot))  
        
V = 5
m = 5
N = 100

xs = rand.randint(1, V, size=(N, m))
ys = to_categorical(xs, num_classes=V)
ys_noisy = np.asarray([[dirichlet_noise(cat, prep_f=lambda v:v*20+1) 
    for cat in s] for s in ys])           
avgs = np.mean(xs, axis=-1).reshape(-1, 1)

#%%

avg_net = AVGPred(V=V, seq_len=m, hidden_size=4)

#%%

avg_net.fit(x=ys_noisy, y=avgs, epochs=2000, verbose=0)

#%%

avg_net.freeze()
avg_net.compile("adam", loss=mean_squared_error)


#%%

sm = SeqModel(seq_len=m, embed_size=5, hidden_size=8, V=V, avg_net=avg_net)


#%%

sm.fit(x=xs, y=[ys, avgs], epochs=10, verbose=1)
    