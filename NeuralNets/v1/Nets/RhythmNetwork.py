# -*- coding: utf-8 -*-

#%%
from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM,\
                 TimeDistributed, Dense, Bidirectional,\
                 Lambda, RepeatVector, Layer
from keras.metrics import categorical_accuracy, mean_absolute_error
from keras.losses import mean_squared_error, categorical_crossentropy
import keras.backend as K
from keras.utils import to_categorical

import numpy as np
import numpy.random as rand

import pickle
from collections import Counter


#%%

class BarEmbedding(Model):
    def __init__(self, V, 
                 note_embed_size, embed_lstm_size, out_size, compile_now=True):
        
        self.embedding_size = out_size
        self.vocab_size = V
        
        embed_layer = Embedding(input_dim=V, output_dim=note_embed_size)
        lstm_layer = Bidirectional(LSTM(embed_lstm_size), merge_mode="concat")
        out_layer = Dense(out_size)

        some_bar = Input(shape=(None,))

        embedded = embed_layer(some_bar)
        bar_processed = lstm_layer(embedded)
        self.bar_embedded = out_layer(bar_processed)
        
        super().__init__(inputs=some_bar, outputs=self.bar_embedded)
        
#        if compile_now:
#            self.compile_default()
#        
#        
#    def compile_default(self):
#        self.compile(optimizer="adam", loss=self.l2_loss)
#
#
#    def l2_loss(self, y_true, y_pred):
#        flat_x_pred = K.flatten(self.bar_embedded)
#        return K.sqrt(K.sum(K.square(flat_x_pred)))
            

class RhythmNetwork(Model):
    def __init__(self, bar_embedder, context_size, enc_lstm_size, dec_lstm_size, compile_now=True):
        prev_bars = [Input(shape=(None,)) for _ in range(context_size)]

        # embed
        prev_bars_embedded = [bar_embedder(pb) for pb in prev_bars]
        embeddings_stacked = Lambda(lambda ls: K.stack(ls, axis=1), 
                           output_shape=(context_size, 
                                         bar_embedder.embedding_size))(prev_bars_embedded)
        
        self.embedded_mat = embeddings_stacked
        
        # encode        
        embeddings_processed = LSTM(enc_lstm_size)(embeddings_stacked)


        # decode
        repeated = Lambda(self._repeat, output_shape=(None, enc_lstm_size))\
                            ([prev_bars[0], embeddings_processed])

        decoded = LSTM(dec_lstm_size, 
                       return_sequences=True, name='dec_lstm')(repeated)

        pred = TimeDistributed(Dense(bar_embedder.vocab_size, activation='softmax'), 
                               name='softmax_layer')(decoded)
    
        super().__init__(inputs=prev_bars, outputs=pred)  
        
        if compile_now:
            self.compile_default()
            
    def compile_default(self):
        self.compile(optimizer="adam", 
                     loss=lambda y_true, y_pred: categorical_crossentropy(y_true, y_pred) + self.l2(), 
                     metrics=[categorical_accuracy])
        
        
    def l2(self):
        squared = K.square(self.embedded_mat)
        summed = K.sum(squared, axis=-1)
        return 1 - K.exp(-K.mean(K.sqrt(summed)))
            
            
    def _repeat(self, args):
        some_bar, vec = args
        bar_len = K.shape(some_bar)[1]
        return RepeatVector(bar_len)(vec)

