# -*- coding: utf-8 -*-

from Nets.MetaPredictor import MetaPredictor
from Nets.CombinedNetwork import CombinedNetwork

#%%

save_dir = "Trainings/meta_no_embed_wrong_loss/"

meta_pred = MetaPredictor.from_saved_custom(save_dir + "meta")

#%%

comb_net = CombinedNetwork.from_saved_custom(save_dir + "weights", meta_pred)