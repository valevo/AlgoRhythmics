# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rand

from keras.utils import to_categorical


import pickle
from Data.utils import label

import os

#%%


class DataGenerator:
    def __init__(self, path):
        self.path = path
        self.num_pieces = None
        
    def load_songs(self):
        files = os.listdir(self.path)
        
        for f in files:
            with open(self.path + "/"  + f, "rb") as handle:
                songs = pickle.load(handle)
                for s in songs:
                    yield s
                    
    def get_num_pieces(self):
        instrument_nums = [song["instruments"] for song in self.load_songs()]
        self.num_pieces = sum(instrument_nums)
        return instrument_nums
            
    
    def get_songs(self, getitem_function, with_metaData=True):
        i = 0
        for song in self.load_songs():
            i += 1
            for i in range(song["instruments"]):
                if with_metaData:
                    yield getitem_function(song[i]), song[i]["metaData"]
                else:
                    yield getitem_function(song[i])
        self.num_pieces = i
        
                    
                    
class RhythmGenerator(DataGenerator):
    def __init__(self, path):
        super().__init__(path)
        song_iter = self.get_rhythms(with_metaData=False)
        label_f, self.label_d = label([beat 
                                  for s in song_iter 
                                  for bar in s 
                                  for beat in bar], start=1)
        self.V = len(self.label_d) + 1
        
    def get_rhythms(self, with_metaData=True):
        yield from self.get_songs(lambda d: d.__getitem__("rhythm"), with_metaData=with_metaData)
        
    def generate_data(self, context_size=1):

        
        song_iter = self.get_rhythms(with_metaData=True)
        while True:
            for rhythms, meta in song_iter:
                bar_len = len(rhythms[0])
                rhythms_labeled = [tuple(self.label_d[b] for b in bar) for bar in rhythms]
                null_bar = (0, )*bar_len
                
                padded_rhythms = [null_bar]*context_size + rhythms_labeled                 
                contexts = [padded_rhythms[i:-(context_size-i)] for i in range(context_size)]
                rhythms_mat = [list(bar) for bar in rhythms_labeled]
                yield ([list(map(np.asarray, contexts)), np.asarray(rhythms_labeled)], 
                       to_categorical(rhythms_mat, num_classes=self.V))
                
            song_iter = self.get_rhythms(with_metaData=True)


class MelodyGenerator(DataGenerator):
    def __init__(self, path):
        super().__init__(path)
        
        song_iter = self.get_notevalues(with_metaData=False)
#        label_f, self.label_d = label([beat 
#                                       for s in song_iter 
#                                       for bar in s 
#                                       for beat in bar], start=1)
        
        self.V = len(set(n for melodies in song_iter 
                         for bar in melodies for n in bar))
#        self.V = len(self.label_d) + 1
        
        
    def get_notevalues(self, with_metaData=True):
        song_iter = self.get_songs(lambda d: d["melody"]["notes"], 
                                  with_metaData=with_metaData)
        
        if with_metaData:
            for melodies, meta in song_iter:
                melodies_None_replaced = [tuple(0 if n is None else n for n in bar) for bar in melodies] 
                yield melodies_None_replaced, meta
        else:
            for melodies in song_iter:
                melodies_None_replaced = [tuple(0 if n is None else n for n in bar) for bar in melodies] 
                yield melodies_None_replaced
        
    def generate_data(self, context_size=1):        
        song_iter = self.get_notevalues(with_metaData=True)
        
        while True:
            for melodies, meta in song_iter:
                bar_len = len(melodies[0])
                null_bar = (0, )*bar_len
                                
                padded_melodies = [null_bar]*context_size + melodies                 
                contexts = [padded_melodies[i:-(context_size-i)] for i in range(context_size)]
                melodies_mat = np.asarray([list(bar) for bar in melodies])
                
                melodies_y = to_categorical(melodies_mat, num_classes=self.V)
                melodies_y[:, :, 0] = 0.
                
                yield (np.transpose(np.asarray(contexts), axes=(1,0,2)), 
                        melodies_y)
                
            song_iter = self.get_notevalues(with_metaData=True)             
            

class CombinedGenerator(DataGenerator):
    def __init__(self, path):
        super().__init__(path)
        self.rhythm_gen = RhythmGenerator(path)
        self.melody_gen = MelodyGenerator(path)
        
        self.rhythm_V = self.rhythm_gen.V
        self.melody_V = self.melody_gen.V
        
    def generate_data(self, rhythm_context_size=1, melody_context_size=1):
        rhythm_iter = self.rhythm_gen.generate_data(rhythm_context_size)
        melody_iter = self.melody_gen.generate_data(melody_context_size)
        
        while True:
            (rhythm_x, rhythms), rhythm_y = next(rhythm_iter)
            melody_x, melody_y = next(melody_iter)
            yield [*rhythm_x, rhythms, melody_x], [rhythm_y, melody_y]
            
            
            
##%%
#
#mg = MelodyGenerator("Data/files")            
#
#m_iter = mg.generate_data()
#
#ns = list(mg.get_notevalues(with_metaData=False))
#
#ns_a = [np.asarray(n_ls) for n_ls in ns]
#
#for n_a in ns_a:
#    n_a[n_a > 0] = 1
#    
#sums = np.asarray([np.sum(n_a, axis=0) for n_a in ns_a])
#
#
##%%
#            
#song_gen = DataGenerator("Data/files")
#
#first = next(song_gen.load_songs())
