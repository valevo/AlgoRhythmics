# -*- coding: utf-8 -*-

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


from Nets.MelodyEncoder import MelodyEncoder


class ChordNetwork(Model):
    def __init__(self, melody_encoder, dense_size, V, compile_now=False):
        m = melody_encoder.m
        
        self.n_voices = 10
        
        root_note = Input(shape=(1, ), name="root_note")
        melody_context = Input(shape=(None, m), name="bar_melody")
        meta_embedded = Input(shape=(self.n_voices, ), name="meta_embedded")
        
        root_encoded = Dense(dense_size)(root_note)
        context_encoded = melody_encoder(melody_context)
        
        inputs_concat = Concat([root_encoded, context_encoded, meta_embedded])
        
        decoded = Dense(dense_size)(inputs_concat)
        
        preds = Dense(V,
                        activation="softmax")(decoded)
        
        super().__init__(inputs=[root_note, melody_context, meta_embedded],
                         outputs=preds)
        
        
        if compile_now:
            self.compile_default()
            
            
    def compile_default(self):
        self.compile("adam",
                     loss=categorical_crossentropy,
                     metrics=[categorical_accuracy])
        

        
        