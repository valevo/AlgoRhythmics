# -*- coding: utf-8 -*-

#%%
from Data.DataGenerators import RhythmGenerator
from Nets.RhythmNetwork import RhythmNetwork

import pickle
import numpy.random as rand
from Data.utils import label


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
rnet = RhythmNetwork(num_categories=len(label_d),
                     embed_size=32,
                     lstm_size=64)
#%%
rnet.fit_generator(r_gen, steps_per_epoch=len(parts), epochs=1)

#%%
rnet.save_weights("rnet_weights.h5")
#%%
rnet2 = RhythmNetwork.from_weights("rnet_weights.h5", num_categories=len(label_d),
                     embed_size=32,
                     lstm_size=64)
#%%
rnet2.fit_generator(r_gen, steps_per_epoch=len(parts), epochs=1)