# -*- coding: utf-8 -*-

# this file needs to be run with the folder "v1" as current working directory
from Nets.RhythmNetwork import RhythmNetwork, BarEmbedding
from Data.DataGenerators import RhythmGenerator

import numpy as np


#%% instatiate neural network

# needs to be returned by RhythmGenerator
V = 4

embed_size = 12
embed_lstm_size = 52
out_size = 6 # there are V**bar_len possible sequences

be = BarEmbedding(V, embed_size, embed_lstm_size, out_size, compile_now=False)

gram_size = 3
lstm_size = 32
dec_lstm_size = 28

rn = RhythmNetwork(be, gram_size, lstm_size, dec_lstm_size)


#%% instantiate data generator

rg = RhythmGenerator("Data/files")

data_iter = rg.generate_data(0, gram_size)


#%% fit

rn.fit_generator(data_iter, steps_per_epoch=300, epochs=20, verbose=2)


#%% check prediction

cur_contexts = next(data_iter)
cur_truth = cur_contexts[1]
cur_contexts = cur_contexts[0]
cur_pred = np.argmax(rn.predict(cur_contexts), axis=-1)

print(np.argmax(cur_truth, axis=-1))
print("\n")
print(cur_pred)





