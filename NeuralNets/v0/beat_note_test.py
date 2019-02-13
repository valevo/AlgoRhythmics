# -*- coding: utf-8 -*-

#%%

from RhythmNetwork import RhythmNetwork
from NoteNetwork import NoteNetwork

from DataGenerators import *


#%% pseudo data

#beats = 



#%%
        
with open("bach_beats.pkl", "rb") as handle:
    data = pickle.load(handle)

data = rand.permutation(data)
parts = [list(map(tuple, part)) for score in data for part in score]
_, label_d = label([b for p in parts for b in p], start=1)
pad_symb = "<s>"
label_d[pad_symb] = 0

#%%
r = RhythmGenerator(4, label_d, "<s>")

r_gen = r.generate_data(parts, shuffle=False)



#%%

bars1000 = [tuple(bar) for _, x in zip(range(1000), r_gen) for bar in x[0][0]]

beats1000 = [b for bar in bars1000 for b in bar]


