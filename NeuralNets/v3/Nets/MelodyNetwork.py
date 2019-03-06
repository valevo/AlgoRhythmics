# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

#%%
from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM,\
                 TimeDistributed, Dense, Bidirectional,\
                 Lambda, RepeatVector, Layer, Conv1D, Reshape
from keras.layers import concatenate as Concat
from keras.metrics import categorical_accuracy, mean_absolute_error
from keras.losses import mean_squared_error, categorical_crossentropy
import keras.backend as K
from keras.utils import to_categorical, plot_model

import numpy as np
import numpy.random as rand

import pickle
from collections import Counter


#%%

class MelodyNetwork(Model):
    def __init__(self, m, V, rhythm_embed_size,
                 conv_f, conv_win_size, enc_lstm_size,
                 dec_lstm_1_size, dec_lstm_2_size, meta_len,
                 compile_now=False):
        
        self.n_voices = 6
        
        prev_melodies = Input(shape=(None, m), name="contexts")
        bar_embedding = Input(shape=(rhythm_embed_size, ), name="bar_rhythm_embedded")
        meta_data = Input(shape=(meta_len,), name="metaData")
        meta_cat = Dense(self.n_voices, activation="softmax")(meta_data)
        

        #encode
        conved = Conv1D(filters=conv_f, kernel_size=conv_win_size)(prev_melodies)
        processed = LSTM(enc_lstm_size)(conved)

        # decode
        proc_rhythm_concat = Concat([processed, bar_embedding, meta_cat])
        proc_repeated = RepeatVector(m)(proc_rhythm_concat)
        
        lstm1_outputs = LSTM(dec_lstm_1_size, return_sequences=True)(proc_repeated)
        lstm2_outputs = LSTM(dec_lstm_2_size, return_sequences=True)(lstm1_outputs)

        preds = TimeDistributed(Dense(V, activation="softmax"))(lstm2_outputs)

        super().__init__(inputs=[prev_melodies, bar_embedding, meta_data], outputs=preds)

        self.params = [m, V, rhythm_embed_size, 
                       conv_f, conv_win_size, 
                       enc_lstm_size, dec_lstm_1_size, dec_lstm_2_size, meta_len]

        if compile_now:
            self.compile_default()
        
    def compile_default(self):
        self.compile("adam", 
                     loss=categorical_crossentropy,
                     metrics=[categorical_accuracy])

        


#%%
#        
#from Data.DataGenerators import MelodyGenerator
#
##%%
#
#mg = MelodyGenerator("Data/files")
#
#m_iter = mg.generate_data()
#
##%%
#        
#V = 4
#l = 7
#
#some_bar = rand.randint(0, V, size=(1, l, 1)) 
#
#some_bar_cat = to_categorical(some_bar, num_classes=V)
#    
#cat_nulled = np.zeros_like(some_bar_cat)
#
#cat_nulled[:] += some_bar_cat[:]
#
#cat_nulled[:, :, 0] = 0.
#
#    
##%%
#
#from keras.losses import categorical_crossentropy        
#
#
##%%
#
#test_inputs = Input(shape=(l, 1))
#
#test_preds = TimeDistributed(Dense(V, activation="softmax"))(test_inputs)
#
#test_model = Model(inputs=test_inputs, outputs=test_preds)
#        
#     
##%%
#   
#test_model.compile("adam", 
#                   loss=categorical_crossentropy,
#                   metrics=[categorical_accuracy])     
#        
#        
##%%        
#        
#test_model.fit(x=some_bar, y=cat_nulled, epochs=10, verbose=2)












        
##%%
#
## input params
#m = 48
#lower, upper = 0, 12
#V = upper - lower
#
#rhythm_emb_size = 32
#
## conv params
#win_size = 3
#f = 4
#
## lstm params
#lstm_size = 52
#
#dec_lstm_1_size = 32
#dec_lstm_2_size = 32
#
##%%
#
#mn = MelodyNetwork(m, V, rhythm_emb_size, f, win_size, lstm_size,
#                   dec_lstm_1_size, dec_lstm_2_size)


#%%

#
#
##%% rhythm data
#
#s_len = 11
#
#
#song_rhythm = rand.random(size=(s_len, rhythm_emb_size))
#
#
##%% melody data
#
#
#
#song_mel = rand.randint(lower, upper, size=(s_len, m))
#
#gram_size = 8
#
#null_bars = np.zeros((gram_size, m)) + upper
#
#padded = np.concatenate([null_bars, song_mel])
#
#batch = np.asarray([padded[i:i+gram_size] for i in range(s_len)])
#
#song_cat = to_categorical(song_mel, num_classes=V)
#
#
##%%
#
#mn.fit(x=[batch, song_rhythm], y=song_cat, epochs=10, verbose=2)
#
##%%
#
#pred = mn.predict([batch, song_rhythm])
#
#
#most_common = [np.bincount(bar).argmax() for bar in song_mel]
#print(most_common)
#print()
#print(song_mel)
#print()
#print(np.argmax(pred, axis=-1))






