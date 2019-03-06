# -*- coding: utf-8 -*-

from keras.layers import Input
from keras.models import Model, load_model
from keras.utils import to_categorical, plot_model

import tensorflow as tf

import numpy as np
import numpy.random as rand


from Nets.RhythmNetwork import BarEmbedding, RhythmNetwork
from Nets.MelodyNetwork import MelodyNetwork

from Data.DataGenerators import CombinedGenerator

#%%

class CombinedNetwork(Model):
    @classmethod
    def from_network_parameters(cls, context_size, melody_bar_len, meta_len,
                                bar_embed_params, rhythm_net_params, melody_net_params, 
                 compile_now=True):
        bar_embedder = BarEmbedding(*bar_embed_params, compile_now=False)
        rhythm_net = RhythmNetwork(bar_embedder, 
                                        *rhythm_net_params, compile_now=False)
        melody_net = MelodyNetwork(*melody_net_params, compile_now=False)
        
        
        combined_net = cls(context_size, melody_bar_len, meta_len,
                           bar_embedder, rhythm_net, melody_net, compile_now=compile_now)
        
        return combined_net
    
    def __init__(self, context_size, melody_bar_len, meta_len,
                               bar_embedder, rhythm_net, melody_net, compile_now=True):
        
        rhythm_contexts = [Input(shape=(None,), 
                                 name="rythm_context_"+str(i)) 
                            for i in range(context_size)]
        rhythms = Input(shape=(None,), name="rhythms")
        melody_contexts = Input(shape=(None, melody_bar_len), name="melody_contexts")
        
        meta = Input(shape=(meta_len,), name="metaData")
        
        
        rhythm_preds = rhythm_net([*rhythm_contexts, meta])
        rhythms_embedded = bar_embedder(rhythms)
        melody_preds = melody_net([melody_contexts, rhythms_embedded, meta])
               
        super().__init__(inputs=[*rhythm_contexts, rhythms, melody_contexts, meta], 
                               outputs=[rhythm_preds, melody_preds])
        
        self.bar_embedder = bar_embedder
        self.rhythm_net = rhythm_net
        self.melody_net = melody_net
        
        if compile_now:
            self.compile_default()
    

            
    def compile_default(self):
        self.compile("adam", 
                       loss="categorical_crossentropy",
                       metrics=["categorical_accuracy"])

##%%
#    
#cg = CombinedGenerator("Data/files")
#cg.get_num_pieces()
#
#rc_size = 3
#data_iter = cg.generate_data(rhythm_context_size=rc_size, melody_context_size=3, 
#                             with_metaData=True)
#
#
##%% PARAMS
#
##rhythm params
#V_rhythm = cg.rhythm_V
#beat_embed_size = 12
#embed_lstm_size = 24
#out_size = 16
#
#context_size = rc_size
#rhythm_enc_lstm_size = 32 
#rhythm_dec_lstm_size = 28
#
#
##melody params
#m = 48
#V_melody = cg.melody_V
#conv_f = 4
#conv_win_size = 3
#melody_enc_lstm_size = 52
#melody_dec_lstm_1_size = 32
#melody_dec_lstm_2_size = 32
#
#meta_data_len = 9
#
##%% INDIVIDUAL NETS
#
#be = BarEmbedding(V=V_rhythm, beat_embed_size=beat_embed_size, embed_lstm_size=embed_lstm_size, out_size=out_size)
#
#rhythm_net = RhythmNetwork(bar_embedder=be, context_size=context_size, 
#                           enc_lstm_size=rhythm_enc_lstm_size, dec_lstm_size=rhythm_dec_lstm_size, meta_len=meta_data_len)
#
#melody_net = MelodyNetwork(m=m, V=V_melody, rhythm_embed_size=out_size,
#                           conv_f=conv_f, conv_win_size=conv_win_size, enc_lstm_size=melody_enc_lstm_size,
#                           dec_lstm_1_size=melody_dec_lstm_1_size, dec_lstm_2_size=melody_dec_lstm_2_size, 
#                           meta_len=meta_data_len)
#
#
##%%
#
#comb_net = CombinedNetwork(context_size, m, meta_data_len, be, rhythm_net, melody_net)
#
#
##%%
#
#from time import asctime
#
#from tensorflow.python.keras.callbacks import TensorBoard
#
##%%
#
#tb = TensorBoard(log_dir="Nets/logs/{}".format(asctime().replace(" ", "_").replace(":", "-")))
#
#
##%%
#comb_net.fit_generator(data_iter, 
#                       steps_per_epoch=cg.num_pieces, epochs=10, verbose=2,
#                       callbacks=[tb])
#
#
#
#
##%%
#
#be.save_weights("bar_embedding_weights")
#rhythm_net.save_weights("rhythm_net_weights")
#melody_net.save("melody_net_weights")
#
#
#
#
#
#
#
#
#
#
#
##%%
#
#be2 = BarEmbedding(V=V_rhythm, beat_embed_size=beat_embed_size, embed_lstm_size=embed_lstm_size, out_size=out_size)
#
#rhythm_net2 = RhythmNetwork(bar_embedder=be2, context_size=context_size, 
#                           enc_lstm_size=rhythm_enc_lstm_size, dec_lstm_size=rhythm_dec_lstm_size, meta_len=meta_data_len)
#
#melody_net2 = MelodyNetwork(m=m, V=V_melody, rhythm_embed_size=out_size,
#                           conv_f=conv_f, conv_win_size=conv_win_size, enc_lstm_size=melody_enc_lstm_size,
#                           dec_lstm_1_size=melody_dec_lstm_1_size, dec_lstm_2_size=melody_dec_lstm_2_size, 
#                           meta_len=meta_data_len)
##%%
#    
#cur = "Wed_Mar__6_17-22-49_2019"
#    
#be2.load_weights("Nets/weights/" + cur + "/bar_embedding_weights")
#rhythm_net2.load_weights("Nets/weights/" + cur + "/rhythm_net_weights")
#melody_net2.load_weights("Nets/weights/" + cur + "/melody_net_weights")
#
#
#
##%%
#
#comb_net2 = CombinedNetwork(context_size, m, meta_data_len, be2, rhythm_net2, melody_net2)
#
#
#
#
##%%
#comb_net2.fit_generator(data_iter, 
#                       steps_per_epoch=cg.num_pieces, epochs=5, verbose=2)
#
#
#
