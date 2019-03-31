# -*- coding: utf-8 -*-

#%%
from Data.DataGeneratorsLead import CombinedGenerator, DataGenerator,\
    MelodyGenerator, RhythmGenerator


#%%


cg = CombinedGenerator("Data/lessfiles", save_conversion_params=False)

gen = cg.generate_data()

#%%
for _ in range(100):
    cur = next(gen)
    print()
    
    
    
#%%
    
mg = MelodyGenerator("Data/lessfiles", save_conversion_params=False)

mgen = mg.get_notevalues_together(with_metaData=False)

#%%



lens = [list(map(len, song)) for song in mgen]


