# -*- coding: utf-8 -*-

from Data.DataGeneratorsLeadMetaChords import ChordGenerator

from Nets.ChordNetwork import ChordNetwork
from Nets.MelodyEncoder import MelodyEncoder

from Nets.MetaEmbedding import MetaEmbedding
from Nets.MetaPredictor import MetaPredictor


import numpy as np


#%% META
meta_embedder = MetaEmbedding.from_saved_custom("Trainings/linear_meta/meta")
meta_embed_size = meta_embedder.embed_size
meta_predictor = MetaPredictor.from_saved_custom("Trainings/linear_meta/meta")
meta_predictor.freeze()



#%%

ch_gen = ChordGenerator("../../Data/music21", save_conversion_params="Trainings/chord_test/",
                        to_list=False, meta_prep_f=meta_embedder.predict)

ch_gen.V

data_iter = ch_gen.generate_forever(batch_size=24)

#x, y = ch_gen.list_data()

#%%
        
menc = MelodyEncoder(m=48, conv_f=4, conv_win_size=1, enc_lstm_size=12)

chn = ChordNetwork(menc, 6, ch_gen.V, compile_now=True)


#%%

chn.fit(x=x, y=y, epochs=2000)

#%%

# ! Number of chords in bar and number of note values
# above 12 don't match !

chn.fit_generator(data_iter, steps_per_epoch=400, epochs=2)
