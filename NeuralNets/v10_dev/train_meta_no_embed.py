# -*- coding: utf-8 -*-

#%%
import numpy as np
import numpy.random as rand

from Data.DataGeneratorsLead import CombinedGenerator

from Nets.MetaPredictor import MetaPredictor, dirichlet_noise
from Nets.MetaEmbedding import get_meta_embedder, MetaEmbedding

from time import asctime
import os

#%%

def gen_meta(comb_gen_instance):
    data_generator = comb_gen_instance.generate_data()
    i = 0
    while True:
        try:
            x, y = next(data_generator)
            yield x[3]
        except IndexError:
            i += 1
            print("IndexError at ", i)
            continue
        except StopIteration:
            return
                
def gen_preds_and_meta(comb_gen_instance,
                       forever=False):
    data_generator = comb_gen_instance.generate_data()
    null = np.zeros((1, 10))
    while True:
        try:
            x, y = next(data_generator)
            meta = x[3]
            rs_one_hot, ms_one_hot = y
            
            rand_alph = rand.randint(2, 10) 
            cur_alphs = lambda v: (v*rand_alph)+1
            rs_noisy = np.asarray([[dirichlet_noise(r_cat, cur_alphs) 
                                    for r_cat in bar] for bar in rs_one_hot])
            ms_noisy = np.asarray([[dirichlet_noise(m_cat, cur_alphs) 
                                    for m_cat in bar] for bar in ms_one_hot])

            embedded = meta
            padded = np.vstack([null, embedded[:-1]])

            yield [rs_noisy, ms_noisy, padded], embedded
            
        except IndexError:
            continue
        except StopIteration:
            if not forever:
                return
            
            data_generator = comb_gen_instance.generate_data()
#%%

if __name__ == "__main__":

#%%
    
    top_dir = "Trainings"
    
    save_dir = asctime().split()
    save_dir = "_".join([*save_dir[0:3], *save_dir[3].split(":")[:2]])
    
    save_dir = "meta_no_embed"
    
    if not os.path.isdir("/".join([top_dir, save_dir, "meta"])):
        os.makedirs("/".join([top_dir, save_dir, "meta"]))
        
    #%%
    cg = CombinedGenerator("../../Data/music21/",
                           save_conversion_params=False,
                           to_list=False)
    
    cg.get_num_pieces()
    
#%%
#    meta_examples = rand.permutation(np.vstack(list(gen_meta(cg))))
#            
#    print("Data set up!")
#        
#    meta_emb, eval_results = get_meta_embedder(meta_examples, 
#                                               embed_size=10, 
#                                               epochs=10, 
#                                               evaluate=True, verbose=1)
#    
#    print("MetaEmbedding trained!\n\tevaluation results:\n\t",
#          eval_results)
    
#%%
    
    pred_meta_gen = gen_preds_and_meta(cg, forever=True)
    
    r_params = (None, cg.rhythm_V)
    m_params = (48, cg.melody_V)
    
    mp = MetaPredictor(r_params, m_params, 10, #meta_embed_size
                       12, 24)
    
#%%
    
    mp.fit_generator(pred_meta_gen, 
                     steps_per_epoch=cg.num_pieces, 
                     epochs=6)
    

#%%

#    meta_emb.save_model_custom("/".join([top_dir, save_dir, "meta"]))

#%%

    mp.save_model_custom("/".join([top_dir, save_dir, "meta"]))




#%%

#meta_embedder = MetaEmbedding.from_saved_custom("Trainings/linear_meta/meta")
#meta_embed_size = meta_embedder.embed_size
#meta_predictor = MetaPredictor.from_saved_custom("Trainings/meta_no_embed_test/meta")
#meta_predictor.freeze()
