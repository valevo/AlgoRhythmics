# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rand

from keras.utils import to_categorical


import pickle
from Data.utils import label

import os

from fractions import Fraction

from itertools import tee

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
                    
                    
    def get_songs_together(self, getitem_function, with_metaData=True):
        for song in self.load_songs():
            num_ins = song["instruments"]
            if with_metaData:
                yield [(getitem_function(song[i]), 
                        song[i]["metaData"]) for i in range(num_ins)]
            else:
                yield [getitem_function(song[i]) for i in range(num_ins)]
            

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


    def generate_forever(self, to_list=False, **generate_params):
        if to_list:
            data_list = list(self.generate_data(**generate_params))
            
            while True:
                yield from data_list
                
        else:
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
        song_iter = self.get_rhythms_together(with_metaData=False)
        label_f, self.label_d = label([beat 
                                  for instruments in song_iter
                                  for s in instruments
                                  for bar in s 
                                  for beat in bar], start=0)
    
        self.null_elem = ()
        self.V = len(self.label_d)
        self.conversion_params["rhythm"] = self.label_d
        if self.save_params_eager:
            self.save_conversion_params()
            
    
    def get_rhythms_together(self, with_metaData=True):
        yield from self.get_songs_together(lambda d: d.__getitem__("rhythm"),
                                           with_metaData=with_metaData)
    
    
    def generate_data(self, context_size=1, rand_stream=None, with_rhythms=True, with_metaData=True):
        song_iter = self.get_rhythms_together(with_metaData=True)
        
        if not rand_stream:
            raise NotImplementedError("Default random stream not implemented!")
        
        for instrument_ls in song_iter:
            cur_i = next(rand_stream)
            print("rhythm rand ind = ", cur_i)
            cur_lead, _ = instrument_ls[cur_i]
            lead_labeled, _ = self.prepare_piece(cur_lead,
                                                             context_size)
            
            for rhythms, meta in instrument_ls:
                rhythms_labeled, context_ls = self.prepare_piece(rhythms, 
                                                               context_size)
                    
                if with_rhythms:
                    context_ls.append(rhythms_labeled)                
                if with_metaData:
                    context_ls.append(self.prepare_metaData(meta, repeat=len(rhythms)))
                    
                context_ls.append(lead_labeled)
                
                yield (context_ls, to_categorical(rhythms_labeled, num_classes=self.V))

         
    def prepare_piece(self, rhythms, context_size):
        bar_len = len(rhythms[0])
        rhythms_labeled = [tuple(self.label_d[b] for b in bar) for bar in rhythms]
        null_bar = (self.label_d[self.null_elem], )*bar_len
                
        padded_rhythms = [null_bar]*context_size + rhythms_labeled                 
        contexts = [padded_rhythms[i:-(context_size-i)] for i in range(context_size)]
        return np.asarray(rhythms_labeled), list(map(np.asarray, contexts))
        
    
    
class MelodyGenerator(DataGenerator):
    def __init__(self, path, save_conversion_params=True):
        super().__init__(path, save_conversion_params=save_conversion_params)
        
        song_iter = self.get_notevalues_together(with_metaData=False)        
        self.V = len(set(n for instruments in song_iter 
                         for melodies in instruments
                         for bar in melodies for n in bar))    
        self.null_elem = 0
        

    
    def get_notevalues_together(self, with_metaData=True):
        song_iter = self.get_songs_together(lambda d: d["melody"]["notes"], 
                                  with_metaData=with_metaData)
        
        if with_metaData:
            for instruments in song_iter:
                cur_ls = []
                for melodies, meta in instruments:
                    melodies_None_replaced = [tuple(0 if n is None else n for n in bar) for bar in melodies] 
                    cur_ls.append((melodies_None_replaced, meta))
                yield cur_ls
        else:
            for instruments in song_iter:
                cur_ls = []
                for melodies in instruments:
                    melodies_None_replaced = [tuple(0 if n is None else n for n in bar) for bar in melodies] 
                    cur_ls.append(melodies_None_replaced)
                yield cur_ls
                
                
    def generate_data(self, context_size=1, rand_stream=None, with_metaData=True):        
        song_iter = self.get_notevalues_together(with_metaData=True)
        
        if not rand_stream:
            raise NotImplementedError("Default random stream not implemented!")
            
        for instrument_ls in song_iter:
            cur_i = next(rand_stream)
            print("melody rand ind = ", cur_i)
            cur_lead, _ = instrument_ls[cur_i]
            lead_mat, _ = self.prepare_piece(cur_lead,
                                             context_size)
        
            for melodies, meta in instrument_ls:
                melodies_mat, contexts = self.prepare_piece(melodies, context_size)
                    
                melodies_y = to_categorical(melodies_mat, num_classes=self.V)
                melodies_y[:, :, 0] = 0.
                    
                
                if with_metaData:
                    yield ([contexts,
                            self.prepare_metaData(meta, repeat=len(melodies)),
                            lead_mat], 
                            melodies_y)
                else:
                    yield ([contexts, lead_mat],
                           melodies_y)
                
                
                
    def prepare_piece(self, melodies, context_size):
        bar_len = len(melodies[0])
        null_bar = (self.null_elem, )*bar_len
        
        melodies_mat = np.asarray([list(bar) for bar in melodies])
        
        padded_melodies = [null_bar]*context_size + melodies                 
        contexts = [padded_melodies[i:-(context_size-i)] for i in range(context_size)]
        contexts = np.transpose(np.asarray(contexts), axes=(1,0,2))
        return melodies_mat, contexts
                


class CombinedGenerator(DataGenerator):
    def __init__(self, path, save_conversion_params=True):
        super().__init__(path, save_conversion_params=save_conversion_params)
        self.rhythm_gen = RhythmGenerator(path, save_conversion_params=save_conversion_params)
        self.melody_gen = MelodyGenerator(path, save_conversion_params=save_conversion_params)
        
        self.rhythm_V = self.rhythm_gen.V
        self.melody_V = self.melody_gen.V
        
    def random_stream(self):
        rhythm_ls = map(len, 
                         self.rhythm_gen.get_rhythms_together(with_metaData=False))
        melody_ls = map(len, 
                         self.melody_gen.get_notevalues_together(with_metaData=False))
        
        for rl, ml in zip(rhythm_ls, melody_ls):
            if not rl == ml:
                raise ValueError("CombinedGenerator.random_stream:\n" + 
                                 "number of instruments in rhythm unequal " + 
                                 "number of instruments in melody!")
                
            yield rand.randint(rl)
                
    def generate_data(self, rhythm_context_size=1, melody_context_size=1, 
                                                  with_metaData=True):
        
#        r1 = self.random_stream()
#        r2 = self.random_stream()
        # OR
        r1, r2 = tee(self.random_stream(), 2)
        
        rhythm_iter = self.rhythm_gen.generate_data(rhythm_context_size,
                                                    rand_stream=r1,
                                                    with_rhythms=True, 
                                                    with_metaData=with_metaData)
        melody_iter = self.melody_gen.generate_data(melody_context_size, 
                                                    rand_stream=r2,
                                                    with_metaData=False)
        
        if with_metaData:
            for (cur_rhythm, cur_melody) in zip(rhythm_iter, melody_iter):
                (*rhythm_x, rhythms, meta, rhythm_lead), rhythm_y = cur_rhythm
                (melody_x, melody_lead), melody_y = cur_melody
                yield ([*rhythm_x, rhythms, melody_x, meta, rhythm_lead, melody_lead], 
                       [rhythm_y, melody_y])
        else:
            for (cur_rhythm, cur_melody) in zip(rhythm_iter, melody_iter):
                (*rhythm_x, rhythms, rhythm_lead), rhythm_y = cur_rhythm
                (melody_x, melody_lead), melody_y = cur_melody
            yield ([*rhythm_x, rhythms, melody_x, rhythm_lead, melody_lead], 
                   [rhythm_y, melody_y])



#%%
            
cg = CombinedGenerator("Data/oldfiles", save_conversion_params=0)


ls = list(cg.generate_data(rhythm_context_size=2, melody_context_size=2, with_metaData=0))         


#%%


rg = RhythmGenerator("Data/oldfiles", save_conversion_params=0)


ls = list(rg.generate_data(context_size=2, with_rhythms=0, with_metaData=0))


#%%

xs, ys = list(zip(*ls))


cs1, cs2 = list(zip(*xs))


#%%

d = DataGenerator("Data/oldfiles", save_conversion_params=False)

f = lambda d: d.__getitem__("rhythm")
rhythms_together = d.get_songs_together(f, with_metaData=False)


#%%

r = RhythmGenerator("Data/oldfiles", save_conversion_params=False)

r_gen = r.generate_data(context_size=2, with_metaData=True)


#%%

m = MelodyGenerator("Data/oldfiles", save_conversion_params=False)

m_gen = m.generate_data(context_size=2, with_metaData=True)





#%%

c = CombinedGenerator("Data/oldfiles", save_conversion_params=0)

c_gen = list(c.generate_data())