# -*- coding: utf-8 -*-

#%%

import numpy as np
import numpy.random as rand

from keras.utils import to_categorical


import pickle
from utils import label

def bar_iter(parts, bar_len, beat_f=lambda x: x):
    for p in parts:
        yield [list(map(beat_f, p[i:i+bar_len])) 
                for i in range(0, len(p), bar_len) 
                if len(p[i:i+bar_len]) == bar_len]



class RhythmGenerator:
    def __init__(self, bar_len, label_dict, padding_symbol, translate_f=None):
        self.bar_len = bar_len
        self.label_d = label_dict
        if not 0 in label_dict.values():
            self.label_d[padding_symbol] = 0
        if translate_f:
            self.translate_f = translate_f
        else:
            self.translate_f = self.label_d.__getitem__
#        self.pad_symb = padding_symbol
        
        self.null_bar = [[self.translate_f(padding_symbol)]*self.bar_len]
        self.n_classes = len(self.label_d)
    
    def generate_data(self, parts, shuffle=True):
        if shuffle:
            parts_shuffled = rand.permutation(parts)
        else:
            parts_shuffled = parts[:]
            
        parts_bars = bar_iter(parts_shuffled, self.bar_len, 
                                              beat_f=self.translate_f)

    
        while True:
            for part in parts_bars:
                prev_bars = np.asarray(self.null_bar + part[:-1])
                cur_padded = np.zeros((len(part), self.bar_len), dtype="int32")
                cur_padded[:, 1:] = np.asarray(part)[:, :-1]
                cur_bars = to_categorical(part, num_classes=self.n_classes)
                yield [prev_bars, cur_padded], cur_bars
            
            parts_shuffled = rand.permutation(parts)
            parts_bars = bar_iter(parts_shuffled, self.bar_len, 
                                  beat_f=self.translate_f)