# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rand

from keras.utils import to_categorical


import pickle
from Data.utils import label

import os

from fractions import Fraction

#%%


class DataGenerator:
    def __init__(self, path, save_conversion_params=True):
        self.path = path
        self.num_pieces = None
        
        self.conversion_params = dict()
        self.save_params_eager = save_conversion_params
        self.params_saved = False
        
    def load_songs(self):
        files = os.listdir(self.path)
        
        for f in files:
            with open(self.path + "/"  + f, "rb") as handle:
                songs = pickle.load(handle)
                for s in songs:
                    yield s

    def get_songs(self, getitem_function, with_metaData=True):
        for song in self.load_songs():
            for i in range(song["instruments"]):
                if with_metaData:
                    yield getitem_function(song[i]), song[i]["metaData"]
                else:
                    yield getitem_function(song[i])

    def get_num_pieces(self):
        instrument_nums = [song["instruments"] for song in self.load_songs()]
        self.num_pieces = sum(instrument_nums)
        return instrument_nums
        
        
        
    def prepare_metaData(self, metaData, repeat=0):
        values = []
        if not "metaData" in self.conversion_params:
            self.conversion_params["metaData"] = sorted(metaData.keys())
            if self.save_params_eager:
                self.save_conversion_params()
        meta_keys = self.conversion_params["metaData"]
        
        if not meta_keys == sorted(metaData.keys()):
            raise ValueError("DataGenerator.prepare_metaData received metaData with different keys!")
            
        
        for k in meta_keys:
            if k == "ts":
                frac = Fraction(metaData[k], _normalize=False)
                values.extend([frac.numerator, frac.denominator])
            else:
                assert isinstance(metaData[k], (float, int))
                values.append(metaData[k])
        
        if not repeat:
            return np.asarray(values, dtype="float")
        else:
            return np.repeat(np.asarray([values], dtype="float"), repeat, axis=0)


    def generate_forever(self, **generate_params):
        data_gen = self.generate_data(**generate_params)
        
        while True:
            yield from data_gen
            data_gen = self.generate_data(**generate_params)


    def save_conversion_params(self, filename=None):
        if not self.conversion_params:
            raise ValueError("DataGenerator.save_conversion_params called while DataGenerator.conversion_params is empty.")
            
        if not filename:
            filename = "DataGenerator.conversion_params"
        
        
        print("CONVERSION PARAMS SAVED TO" + " Data/" + filename)
        
        with open("Data/" + filename, "wb") as handle:
            pickle.dump(self.conversion_params, handle)
            
                    
class RhythmGenerator(DataGenerator):
    def __init__(self, path, save_conversion_params=True):
        super().__init__(path, save_conversion_params=save_conversion_params)
        song_iter = self.get_rhythms(with_metaData=False)
        label_f, self.label_d = label([beat 
                                  for s in song_iter 
                                  for bar in s 
                                  for beat in bar], start=1)
        self.V = len(self.label_d) + 1
        self.conversion_params["rhythm"] = self.label_d
        if self.save_params_eager:
            self.save_conversion_params()
        
    def get_rhythms(self, with_metaData=True):
        yield from self.get_songs(lambda d: d.__getitem__("rhythm"), with_metaData=with_metaData)
        
        
    def generate_data(self, context_size=1, with_rhythms=True, with_metaData=True):
        song_iter = self.get_rhythms(with_metaData=True)
        for rhythms, meta in song_iter:
            bar_len = len(rhythms[0])
            rhythms_labeled = [tuple(self.label_d[b] for b in bar) for bar in rhythms]
            null_bar = (0, )*bar_len
                
            padded_rhythms = [null_bar]*context_size + rhythms_labeled                 
            contexts = [padded_rhythms[i:-(context_size-i)] for i in range(context_size)]
            rhythms_mat = [list(bar) for bar in rhythms_labeled]
                
            x_ls = list(map(np.asarray, contexts))
            if with_rhythms:
                x_ls.append(np.asarray(rhythms_labeled))                
            if with_metaData:
                x_ls.append(self.prepare_metaData(meta, repeat=len(rhythms)))
            
            yield (x_ls, to_categorical(rhythms_mat, num_classes=self.V))

         
            
class MelodyGenerator(DataGenerator):
    def __init__(self, path, save_conversion_params=True):
        super().__init__(path, save_conversion_params=save_conversion_params)
        
        song_iter = self.get_notevalues(with_metaData=False)        
        self.V = len(set(n for melodies in song_iter 
                         for bar in melodies for n in bar))        
        
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
    
    
    def generate_data(self, context_size=1, with_metaData=True):        
        song_iter = self.get_notevalues(with_metaData=True)
        for melodies, meta in song_iter:
            bar_len = len(melodies[0])
            null_bar = (0, )*bar_len
                                
            padded_melodies = [null_bar]*context_size + melodies                 
            contexts = [padded_melodies[i:-(context_size-i)] for i in range(context_size)]
            melodies_mat = np.asarray([list(bar) for bar in melodies])
                
            melodies_y = to_categorical(melodies_mat, num_classes=self.V)
            melodies_y[:, :, 0] = 0.
                
            if with_metaData:
                yield ([np.transpose(np.asarray(contexts), axes=(1,0,2)),
                        self.prepare_metaData(meta, repeat=len(melodies))], 
                        melodies_y)
            else:
                yield (np.transpose(np.asarray(contexts), axes=(1,0,2)), 
                       melodies_y)
                


class CombinedGenerator(DataGenerator):
    def __init__(self, path, save_conversion_params=True):
        super().__init__(path, save_conversion_params=save_conversion_params)
        self.rhythm_gen = RhythmGenerator(path, save_conversion_params=save_conversion_params)
        self.melody_gen = MelodyGenerator(path, save_conversion_params=save_conversion_params)
        
        self.rhythm_V = self.rhythm_gen.V
        self.melody_V = self.melody_gen.V
        
    def generate_data(self, rhythm_context_size=1, melody_context_size=1, 
                                                  with_metaData=True):
        
        rhythm_iter = self.rhythm_gen.generate_data(rhythm_context_size,
                                                    with_rhythms=True, 
                                                    with_metaData=with_metaData)
        melody_iter = self.melody_gen.generate_data(melody_context_size, 
                                                    with_metaData=False)
        
        if with_metaData:
            (*rhythm_x, rhythms, meta), rhythm_y = next(rhythm_iter)
            melody_x, melody_y = next(melody_iter)
            yield [*rhythm_x, rhythms, melody_x, meta], [rhythm_y, melody_y]
        else:
            (*rhythm_x, rhythms), rhythm_y = next(rhythm_iter)
            melody_x, melody_y = next(melody_iter)
            yield [*rhythm_x, rhythms, melody_x], [rhythm_y, melody_y]



#%%
            
cg = CombinedGenerator("Data/oldfiles", save_conversion_params=0)


ls = list(cg.generate_data(rhythm_context_size=2, melody_context_size=2, with_metaData=0))         


#%%


rg = RhythmGenerator("Data/oldfiles", save_conversion_params=0)


ls = list(rg.generate_data(context_size=2, with_rhythms=0, with_metaData=0))


#%%

xs, ys = list(zip(*ls))


cs1, cs2 = list(zip(*xs))


         