# -*- coding: utf-8 -*-

#%%

import numpy as np
import numpy.random as rand

from keras.utils import to_categorical


import pickle
from Data.utils import label

import os

#%%

class RhythmGenerator:
    def __init__(self, path):
        self.path = path      

        self.all_songs = list(self.load_data())
        
        self.instrument_ids = set(ins["id"] for s in self.all_songs 
                                  for ins in s["instruments"])
        
    def load_data(self):
        files = os.listdir(self.path)
        
        for f in files:
            with open(self.path + "/"  + f, "rb") as handle:
                songs = pickle.load(handle)
                for s in songs:
                    yield s
                    
    def get_songs(self, instrument_id, with_metaData=True):
        for song in self.all_songs:
            if len(song["instruments"]) <= instrument_id: 
                continue
            
            cur_ins = song["instruments"][instrument_id]
            if with_metaData:
                yield cur_ins["rhythm"], cur_ins["metaData"]
            else:
                yield cur_ins["rhythm"]


    

    def generate_data(self, instrument_id, n_grams=1):
        song_iter = self.get_songs(instrument_id, with_metaData=False)

        label_f, label_d = label([beat for s in song_iter for bar in s for beat in bar], start=1)
        V = len(label_d) + 1
        
        song_iter = self.get_songs(instrument_id, with_metaData=True)
        while True:
            for rhythms, meta in song_iter:                
                bar_len = len(rhythms[0])
                rhythms_labeled = [tuple(label_d[b] for b in bar) for bar in rhythms]
                null_bar = (0, )*bar_len
                
                padded_rhythms = [null_bar]*n_grams + rhythms_labeled                 
                contexts = [padded_rhythms[i:-(n_grams-i)] for i in range(n_grams)]
                rhythms_mat = [list(bar) for bar in rhythms_labeled]
                yield list(map(np.asarray, contexts)), to_categorical(rhythms_mat, num_classes=V)
                
            song_iter = self.get_songs(instrument_id, with_metaData=True)
            
            
#%%
            
rg = RhythmGenerator("Data/files")

di = rg.generate_data(0, n_grams=3)





