# -*- coding: utf-8 -*-

from Data.DataGeneratorsLeadMeta import CombinedGenerator

from Nets.MetaEmbedding import MetaEmbedding
from Nets.MetaPredictor import MetaPredictor

from Nets.RhythmEncoder import BarEmbedding, RhythmEncoder
from Nets.RhythmNetwork import RhythmNetwork
from Nets.MelodyEncoder import MelodyEncoder
from Nets.MelodyNetwork import MelodyNetwork
from Nets.CombinedNetwork import CombinedNetwork



from time import asctime
import os

from tensorflow.python.keras.callbacks import TensorBoard

def without_lead(cg_inst, rc, mc):
    data_gen = cg.generate_data(rhythm_context_size=rc,
                            melody_context_size=mc,
                            with_metaData=True)
    
    while True:
        for x, y in data_gen:
            yield (x[:-2], y)
        
        data_gen = cg.generate_data(rhythm_context_size=rc,
                            melody_context_size=mc,
                            with_metaData=True)


if __name__ == "__main__":

    num_epochs = 10
    j = 1   # checkpoint frequency
    
    
    # META
    meta_embedder = MetaEmbedding.from_saved_custom("meta_saved")
    meta_embed_size = meta_embedder.embed_size
    meta_predictor = MetaPredictor.from_saved_custom("meta_saved")
    meta_predictor.freeze()

    # CHANGE
    music_dir = "../../Data/music21/"
    cg = CombinedGenerator(music_dir, save_conversion_params=1,
                           to_list=0, meta_prep_f=meta_embedder.predict)
    cg.get_num_pieces()
    
    rc_size = 4
    mc_size = 4
#    data_iter = cg.generate_forever(rhythm_context_size=rc_size, 
#                                    melody_context_size=mc_size, 
#                                 with_metaData=True, to_list=False)
    
    data_iter = without_lead(cg, rc_size, mc_size)
    print("\nData generator set up...\n")
    
    
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
    melody_dec_lstm_size = 32
    
    meta_data_len = 10
    
    # INDIVIDUAL NETS
    
    bar_embedder = BarEmbedding(V=V_rhythm, beat_embed_size=beat_embed_size, 
                                embed_lstm_size=embed_lstm_size, 
                                out_size=out_size)
    rhythm_encoder = RhythmEncoder(bar_embedder=bar_embedder,
                                   context_size=rc_size,
                                   lstm_size=rhythm_enc_lstm_size)
    rhythm_net = RhythmNetwork(rhythm_encoder=rhythm_encoder,
                               dec_lstm_size=rhythm_dec_lstm_size, V=V_rhythm, 
                               dec_use_meta=True, compile_now=True)
    

    # ATTENTION: conv_win_size must not be greater than context size!
    melody_encoder = MelodyEncoder(m=m, conv_f=conv_f, conv_win_size=min(mc_size, conv_win_size), 
                                   enc_lstm_size=melody_enc_lstm_size)
    melody_net = MelodyNetwork(melody_encoder=melody_encoder, 
                               rhythm_embed_size=out_size,
                               dec_lstm_size=melody_dec_lstm_size, V=V_melody,
                               dec_use_meta=True, compile_now=True)

    
    print("Individual networks set up...\n")
    
    
    #
    comb_net = CombinedNetwork(context_size, m, meta_embed_size, 
                               bar_embedder, rhythm_net, melody_net, meta_predictor,
                               generation=False, compile_now=True)
    
    print("Combined network set up...\n")
    
    #
    top_dir = "Nets/"
    log_dir = "logs/"
    weight_dir = "weights/"
    
    cur_date_time = asctime().replace(" ", "_").replace(":", "-")
    tb = TensorBoard(log_dir= top_dir + log_dir + cur_date_time)
    os.makedirs(top_dir + weight_dir + cur_date_time)
    
    
    #
    for cur_iteration in range(int(num_epochs/j)):
    
        print("\nITERATION ", cur_iteration)
        
        comb_net.fit_generator(data_iter, 
                               steps_per_epoch=cg.num_pieces, 
                               epochs=cur_iteration*j+j, 
                               initial_epoch=cur_iteration*j,
                               verbose=2, callbacks=[tb])
    
        cur_folder_name = cur_date_time + "/_checkpoint_" + str(cur_iteration)
        os.makedirs(top_dir + weight_dir + cur_folder_name)
        comb_net.save_model_custom(top_dir + weight_dir + cur_folder_name)

        
    
    
    comb_net.save_model_custom(top_dir + weight_dir + cur_date_time)
