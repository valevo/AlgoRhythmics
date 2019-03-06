# -*- coding: utf-8 -*-

from Data.DataGenerators import CombinedGenerator

from Nets.RhythmNetwork import BarEmbedding, RhythmNetwork
from Nets.MelodyNetwork import MelodyNetwork
from Nets.CombinedNetwork import CombinedNetwork



from time import asctime
import os

from tensorflow.python.keras.callbacks import TensorBoard



if __name__ == "__main__":

    num_epochs = 2   
        
        
    #   
    cg = CombinedGenerator("Data/files")
    cg.get_num_pieces()
    
    rc_size = 3
    data_iter = cg.generate_data(rhythm_context_size=rc_size, melody_context_size=3, 
                                 with_metaData=True)
    print("Data generator set up...\n")
    
    # PARAMS
    
    #rhythm params
    V_rhythm = cg.rhythm_V
    beat_embed_size = 12
    embed_lstm_size = 24
    out_size = 16
    
    context_size = rc_size
    rhythm_enc_lstm_size = 32 
    rhythm_dec_lstm_size = 28
    
    
    #melody params
    m = 48
    V_melody = cg.melody_V
    conv_f = 4
    conv_win_size = 3
    melody_enc_lstm_size = 52
    melody_dec_lstm_1_size = 32
    melody_dec_lstm_2_size = 32
    
    meta_data_len = 9
    
    # INDIVIDUAL NETS
    be = BarEmbedding(V=V_rhythm, beat_embed_size=beat_embed_size, embed_lstm_size=embed_lstm_size, out_size=out_size)
    
    rhythm_net = RhythmNetwork(bar_embedder=be, context_size=context_size, 
                               enc_lstm_size=rhythm_enc_lstm_size, dec_lstm_size=rhythm_dec_lstm_size, meta_len=meta_data_len)
    
    melody_net = MelodyNetwork(m=m, V=V_melody, rhythm_embed_size=out_size,
                               conv_f=conv_f, conv_win_size=conv_win_size, enc_lstm_size=melody_enc_lstm_size,
                               dec_lstm_1_size=melody_dec_lstm_1_size, dec_lstm_2_size=melody_dec_lstm_2_size, 
                               meta_len=meta_data_len)
    
    print("Individual networks set up...\n")
    
    
    #
    comb_net = CombinedNetwork(context_size, m, meta_data_len, be, rhythm_net, melody_net)
    
    print("Combined network set up...\n")
    
    #
    top_dir = "Nets/"
    log_dir = "logs/"
    weight_dir = "weights/"
    
    cur_date_time = asctime().replace(" ", "_").replace(":", "-")
    tb = TensorBoard(log_dir= top_dir + log_dir + cur_date_time)
    os.makedirs(top_dir + weight_dir + cur_date_time)
    
    
    #
    comb_net.fit_generator(data_iter, 
                           steps_per_epoch=cg.num_pieces, 
                           epochs=num_epochs, 
                           verbose=2,
                           callbacks=[tb])
    
    
    be.save_weights(top_dir + weight_dir + cur_date_time + "/bar_embedding_weights")
    rhythm_net.save_weights(top_dir + weight_dir + cur_date_time + "/rhythm_net_weights")
    melody_net.save(top_dir + weight_dir + cur_date_time +"/melody_net_weights")