# -*- coding: utf-8 -*-

#%%
from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM,\
                 TimeDistributed, Dense, Bidirectional,\
                 Lambda, RepeatVector, Layer, Conv1D, Reshape
from keras.metrics import categorical_accuracy, mean_absolute_error
from keras.losses import mean_squared_error, categorical_crossentropy
import keras.backend as K
from keras.utils import to_categorical

import numpy as np
import numpy.random as rand

import pickle
from collections import Counter

#%% "deconvolution"

## input params
#m = 48
#lower, upper = 0, 12
#V = upper - lower
#
## conv params
#win_size = 3
#f = 4
#
##lstm params
#factor = m+3
#lstm_size = 24*factor
#
##deconv params
#deconv_f = V+5
#deconv_win_size = 1
#
#inputs = Input(shape=(None, m))
#conved = Conv1D(filters=f, kernel_size=win_size)(inputs)
#processed = LSTM(lstm_size)(conved)
#reshaped = Reshape(target_shape=(factor, int(lstm_size/factor)))(processed)
#deconved = Conv1D(filters=deconv_f, kernel_size=4)(reshaped)
#preds = TimeDistributed(Dense(V, activation="softmax"))(deconved)
#model = Model(inputs=inputs, outputs=preds)


#%%

# input params
m = 48
lower, upper = 0, 12
V = upper - lower

# conv params
win_size = 3
f = 4

# lstm params
lstm_size = 52


inputs = Input(shape=(None, m))

conved = Conv1D(filters=f, kernel_size=win_size)(inputs)

#shape: (lstm_size, )
processed = LSTM(lstm_size)(conved)


#%% TRANSFORMER

# transformer params
dense1_size = 32
dense2_size = 32
dense3_size = 32

#shape: (m, lstm_size)
proc_repeated = RepeatVector(m)(processed)

dense1 = TimeDistributed(Dense(dense1_size))(proc_repeated)
dense2 = TimeDistributed(Dense(dense2_size))(dense1)
dense3 = TimeDistributed(Dense(dense3_size))(dense2)

preds = TimeDistributed(Dense(V, activation="softmax"))(dense3)


model = Model(inputs=inputs, outputs=preds)



#%%

model.compile("adam", 
              loss=categorical_crossentropy,
              metrics=[categorical_accuracy])

#%%

s_len = 11


song = rand.randint(lower, upper, size=(s_len, m))

gram_size = 8

null_bars = np.zeros((gram_size, m)) + upper

padded = np.concatenate([null_bars, song])

batch = np.asarray([padded[i:i+gram_size] for i in range(s_len)])


song_cat = to_categorical(song, num_classes=V)

#batch = np.asarray([padded[i:-(gram_size-i)] for i in range(gram_size)])


#%%

model.fit(x=batch, y=song_cat, epochs=10, verbose=1)

#%%



pred = model.predict(batch)



#%%

a =  rand.randint(1, 6, (6,))

print(a)

a.reshape((2, 3))




#%%

k = 3
ins = Input(shape=(k, ))

rep = RepeatVector(2)(ins)

test = Model(inputs=ins, outputs=rep)





