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
        meta_repeated = Lambda(self._repeat, 
                               output_shape=(None, self.n_voices)
                               )([conved, meta_cat])
        
        
        conved_with_meta = Concat([conved, meta_repeated], axis=-1)
        processed = LSTM(enc_lstm_size)(conved_with_meta)

        # decode
        proc_rhythm_concat = Concat([processed, bar_embedding])
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



    def _repeat(self, args):
        conv_mat, vec = args
        conv_len = K.shape(conv_mat)[1]
        return RepeatVector(conv_len)(vec)
        

