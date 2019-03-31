# -*- coding: utf-8 -*-

#%%


from keras.layers import Dense, Input, LSTM, TimeDistributed, Lambda, Concatenate,\
                Bidirectional, Dropout
from keras.models import Model

from keras.metrics import categorical_crossentropy, mean_absolute_error,\
                mean_squared_error
                
from keras.regularizers import Regularizer

import keras.backend as K
from keras.utils import to_categorical


import numpy as np
import numpy.random as rand

from meta_embed import MetaEmbedding, get_meta_embedder

import json 

#%%

# meta data is rolling average!
#  => MetaPredictor needs to be function of both (rythm, melody) and
#  previous metaData

class MetaPredictor(Model):
    def __init__(self, rhythm_params, melody_params, meta_embed_size, 
                 lstm_size, dense_size, compile_now=True):
        
        rhythms_dist = Input(shape=rhythm_params)
        melodies_dist = Input(shape=melody_params)
        prev_meta = Input(shape=(meta_embed_size, ))
        
        prev_meta_dropped = Dropout(0.5)(prev_meta)
        
        rhythms_processed = Bidirectional(LSTM(lstm_size), merge_mode="concat")(rhythms_dist)
        melodies_processed = Bidirectional(LSTM(lstm_size), merge_mode="concat")(melodies_dist)
        
        processed_concat = Concatenate()([rhythms_processed, melodies_processed])
        
        pre_meta = Dense(dense_size)(processed_concat)
        
        metas_combined = Concatenate()([prev_meta_dropped, pre_meta])
        
        meta_embedded = Dense(meta_embed_size, 
                              activation="softmax")(metas_combined)
        
        
        super().__init__(inputs=[rhythms_dist, melodies_dist, prev_meta],
                         outputs=meta_embedded,
                         name=repr(self))
        
        self.params = {"rhythm_params": rhythm_params,
                       "melody_params": melody_params,
                       "meta_embed_size": meta_embed_size,
                       "lstm_size": lstm_size,
                       "dense_size": dense_size}

        
        if compile_now:
            self.compile_default()
            
            
    def compile_default(self):
        self.compile("adam",
                     loss=categorical_crossentropy,
                     metrics=[mean_absolute_error])
    
    def __repr__(self):
        return "MetaPredictor"
        
    
    def save_model_custom(self, dir_to_save):
        self.save_weights(dir_to_save + "/meta_predictor_weights")
        with open(dir_to_save + "/meta_predictor_parameters.json", "w") as handle:
            json.dump(self.params, handle)
            
    @classmethod  
    def from_saved_custom(cls, save_dir, compile_now=False):            
        with open(save_dir + "/meta_predictor_parameters.json", "r") as handle:
            param_dict = json.load(handle)
            
        meta_pred = cls(param_dict["rhythm_params"],
                            param_dict["melody_params"],
                            param_dict["meta_embed_size"],
                            param_dict["lstm_size"],
                            param_dict["dense_size"],
                            compile_now=compile_now)
        
        meta_pred.load_weights(save_dir + "/meta_predictor_weights")
        
        return meta_pred
    
#%%

def dirichlet_noise(one_hot, prep_f=lambda v: v*10+1):
    return rand.dirichlet(prep_f(one_hot))        



##%%
#
#def rolling_f(data, f):
#    to_fill = np.zeros((data.shape[0], ))
#    to_fill[0] = f(data[0])
#    for i, row in enumerate(data[1:], start=1):
#        to_fill[i] = f([to_fill[i-1], f(row)])
#    
#    return to_fill
#        
#
##%%
#r_bar_len = 4        
#m_bar_len = 48
#N = 100
#song_len = 10
#
#V_rhythm =  21
#V_melody = 11
#
#test_rhythms = rand.randint(1, V_rhythm, size=(N, song_len, r_bar_len))
#test_melodies = rand.randint(1, V_melody, size=(N, song_len, m_bar_len))
#
##%%
#
#test_meta = np.asarray([np.transpose([rolling_f(s, np.mean),
#                         rolling_f(s, np.max),
#                         rolling_f(s, np.min)]) for s in test_rhythms])
#
#
#
###%%
##
##test_meta = np.transpose([np.mean(test_rhythms, axis=-1), 
##                          np.var(test_melodies, axis=-1),
##                          np.min(test_rhythms, axis=-1),
##                          np.max(test_melodies, axis=-1)])
##    
#
#
##%%
#    
#meta_examples = test_meta.reshape((N*song_len, -1))
#    
#meta_emb, eval_results = get_meta_embedder(meta_examples, 4, epochs=3000, evaluate=True)
#
#
#
#
##%%
#
#meta_embedded = np.asarray([meta_emb.predict(s) for s in test_meta])
#
#
#rhythms_cat = to_categorical(test_rhythms, num_classes=V_rhythm)
#melodies_cat = to_categorical(test_melodies, num_classes=V_melody)
#
##%%
#
#noisy_rhythms_cat = np.asarray([[[dirichlet_noise(r_cat) for r_cat in bar] 
#            for bar in s] for s in rhythms_cat])
#noisy_melodies_cat = np.asarray([[[dirichlet_noise(m_cat) for m_cat in bar] 
#            for bar in s] for s in melodies_cat])
#    
##%%
#    
#meta_pred = MetaPredictor(rhythm_params=(r_bar_len, V_rhythm), 
#                          melody_params=(m_bar_len, V_melody),
#                          meta_embedder=meta_emb)
#
#
##%%
#
#def train_gen():
#    
#    
#    while True:
#        for r, m, emb_meta in zip(noisy_rhythms_cat, noisy_melodies_cat, meta_embedded):
#            emb_meta_padded = np.vstack((np.zeros_like(emb_meta[0]), emb_meta))[:-1]
#            yield [r, m, emb_meta_padded], emb_meta
#
##meta_pred.fit(x=[noisy_rhythms_cat, noisy_melodies_cat], 
##              y=meta_embedded,
##              epochs=100)
#
#g = train_gen()
#
##%%
#
#meta_pred.fit_generator(g, steps_per_epoch=N, epochs=100)
#
#
##%%
#
#rand_i = rand.randint(N)
#
#null_meta_emb = np.zeros((meta_emb.embed_size))
#
#pred_meta_emb = meta_pred.predict([
#        noisy_rhythms_cat[rand_i],
#        noisy_melodies_cat[rand_i],
#        np.vstack((null_meta_emb, meta_embedded[rand_i]))[:-1]
#        ])
#
#print(rand_i, "\n")
#print(pred_meta_emb, "\n")
#print(meta_emb.predict(test_meta[rand_i]), "\n")
#print(test_meta[rand_i], "\n")
