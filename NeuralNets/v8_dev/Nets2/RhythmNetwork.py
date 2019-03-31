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


from RhythmEncoder import BarEmbedding, RhythmEncoder

#%%
        

class RhythmNetwork(Model):
    def __init__(self, bar_embedder, rhythm_encoder, dec_lstm_size, 
                 enc_use_meta=False, dec_use_meta=False, compile_now=False):
        self.n_voices = 6

        context_size = rhythm_encoder.context_size

        prev_bars = [Input(shape=(None,), name="context_" + str(i)) 
                            for i in range(context_size)]
        
        if enc_use_meta or dec_use_meta:
            meta_cat = Input(shape=(None,), name="metaData")

                        
        # encode        
        embeddings_processed = rhythm_encoder(prev_bars)

        
        # decode
        if dec_use_meta:
            encoded_size += self.n_voices
            embeddings_processed = Concat([embeddings_processed, meta_cat])
        
        repeated = Lambda(self._repeat, output_shape=(None, encoded_size))\
                                ([prev_bars[0], embeddings_processed])

        decoded = LSTM(dec_lstm_size, 
                       return_sequences=True, name='dec_lstm')(repeated)

        preds = TimeDistributed(Dense(bar_embedder.vocab_size, activation='softmax'), 
                               name='softmax_layer')(decoded)
    
        if enc_use_meta or dec_use_meta:
            super().__init__(inputs=[*prev_bars, meta_cat], outputs=preds)  
        else:
            super().__init__(inputs=prev_bars, outputs=preds)  
            
        self.params = [context_size, dec_lstm_size,
                       enc_use_meta, dec_use_meta]
        
        self.use_meta = enc_use_meta or dec_use_meta

        
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
    
    
#%%
#        
#V = 5
#
#test_data = rand.randint(V, size=(3, 4))
#
##%%
#
#be = BarEmbedding(V, beat_embed_size=3, embed_lstm_size=2, out_size=2)
#
#
##%%
#
#rn = RhythmNetwork(be, context_size=1, enc_lstm_size=2, dec_lstm_size=2, 
#                 enc_use_meta=True, dec_use_meta=True, compile_now=False)

