import numpy as np
import numpy.random as rand

from Data.DataGeneratorsLead import CombinedGenerator as CGLead
from Data.DataGenerators import CombinedGenerator

from Nets.CombinedNetwork2 import BarEmbedding, RhythmEncoder, RhythmNetwork,\
                                    MelodyEncoder, MelodyNetwork,\
                                    CombinedNetwork
                                    
from Nets.MetaEmbedding import MetaEmbedding
from Nets.MetaPredictor import MetaPredictor

#%%

cg = CombinedGenerator("Data/lessfiles", save_conversion_params=False)

data_gen = cg.generate_data(rhythm_context_size=1,
                            melody_context_size=1,
                            with_metaData=True)

#%% META
meta_embedder = MetaEmbedding.from_saved_custom("meta_saved")
meta_embed_size = meta_embedder.embed_size
meta_predictor = MetaPredictor.from_saved_custom("meta_saved")
meta_predictor.freeze()

#%% RHYTHM
rc = 4
rV = cg.rhythm_V
r_embed_size = 10

bar_embedder = BarEmbedding(V=rV, beat_embed_size=12, 
                            embed_lstm_size=14, out_size=r_embed_size)
rhythm_encoder = RhythmEncoder(bar_embedder=bar_embedder,
                               context_size=rc,
                               lstm_size=18)
rhythm_net = RhythmNetwork(rhythm_encoder=rhythm_encoder,
                           dec_lstm_size=18, V=rV, 
                           dec_use_meta=True, compile_now=True)

#%% MELODY

mc = 4
mV = cg.melody_V
m = 48

melody_encoder = MelodyEncoder(m=m, conv_f=4, conv_win_size=3, enc_lstm_size=16)
melody_net = MelodyNetwork(melody_encoder=melody_encoder, rhythm_embed_size=r_embed_size,
                           dec_lstm_size=16, V=mV,
                           dec_use_meta=True)


#%% COMBINED

combined_net = CombinedNetwork(context_size=rc, melody_bar_len=m,
                               meta_embed_size=meta_embed_size, 
                               bar_embedder=bar_embedder, rhythm_net=rhythm_net, 
                               melody_net=melody_net, meta_net=meta_predictor)


#%% TRAIN







#%% TODO
#   MetaPredictor currently has 17,000 parameters -> cut by half
#   RhythmEncoder & Network: init_with_Encoder needs to initialise BarEmbedding as well
#   CombinedNetwork: fix save_custom and from_saved_custom
