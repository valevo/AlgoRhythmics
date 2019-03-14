# -*- coding: utf-8 -*-

#%%
from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM,\
                 TimeDistributed, Dense, Bidirectional,\
                 Lambda, RepeatVector, Layer, Concatenate
from keras.layers import concatenate as Concat
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
                 beat_embed_size, embed_lstm_size, out_size, compile_now=False):
                
        self.embedding_size = out_size
        self.vocab_size = V
        
        embed_layer = Embedding(input_dim=V, output_dim=beat_embed_size)
        lstm_layer = Bidirectional(LSTM(embed_lstm_size), merge_mode="concat")
        out_layer = Dense(out_size)

        some_bar = Input(shape=(None,))

        embedded = embed_layer(some_bar)
        bar_processed = lstm_layer(embedded)
        self.bar_embedded = out_layer(bar_processed)
        
        super().__init__(inputs=some_bar, outputs=self.bar_embedded)
        
        self.params = [V, beat_embed_size, embed_lstm_size, out_size]
        

class RhythmNetwork(Model):
    def __init__(self, bar_embedder, context_size, 
                 enc_lstm_size, dec_lstm_size, meta_len, compile_now=False):
        self.num_calls = 0
        self.n_voices = 6


        prev_bars = [Input(shape=(None,), name="context_" + str(i)) 
                            for i in range(context_size)]
        meta_data = Input(shape=(meta_len,), name="metaData")
        meta_cat = Dense(self.n_voices, activation="softmax")(meta_data)
        
        # embed
        embeddings_with_meta = [Concat([bar_embedder(pb), meta_cat]) for pb in prev_bars]
        embeddings_stacked = Lambda(lambda ls: K.stack(ls, axis=1), 
                           output_shape=(context_size, 
                                         bar_embedder.embedding_size + self.n_voices)
                           )(embeddings_with_meta)
                        
        # encode        
        embeddings_processed = LSTM(enc_lstm_size)(embeddings_stacked)


        # decode        
        repeated = Lambda(self._repeat, output_shape=(None, enc_lstm_size))\
                            ([prev_bars[0], embeddings_processed])

        decoded = LSTM(dec_lstm_size, 
                       return_sequences=True, name='dec_lstm')(repeated)

        pred = TimeDistributed(Dense(bar_embedder.vocab_size, activation='softmax'), 
                               name='softmax_layer')(decoded)
    
        super().__init__(inputs=[*prev_bars, meta_data], outputs=pred)  
        
        self.params = [context_size, enc_lstm_size, dec_lstm_size, meta_len]
        
        if compile_now:
            self.compile_default()
            
    def compile_default(self):
        self.compile(optimizer="adam", 
                     loss=lambda y_true, y_pred: categorical_crossentropy(y_true, y_pred),# + 10000.0, #self.l2(y_true, y_pred), 
                     metrics=[categorical_accuracy])
            
            
    def _repeat(self, args):
        some_bar, vec = args
        bar_len = K.shape(some_bar)[-1]
        return RepeatVector(bar_len)(vec)