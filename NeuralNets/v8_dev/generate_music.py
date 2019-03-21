# -*- coding: utf-8 -*-

from Nets.CombinedNetwork import CombinedNetwork

import numpy as np
import numpy.random as rand


if __name__ == "__main__":
    
    #%%
    
    weights_folder = "Nets/weights/test_net"
    
    comb_net = CombinedNetwork.from_saved_custom(weights_folder, 
                                                 generation=True,
                                                 compile_now=False)
    
    context_size = comb_net.params["context_size"]
    
    V_rhythm = comb_net.params["bar_embed_params"][0]
    
    m, V_melody = comb_net.params["melody_net_params"][0], comb_net.params["melody_net_params"][1]
    
    meta_len = comb_net.params["meta_len"]
    
    print("\n", "-"*40,  "\nINFO FOR LOADED NET:", comb_net)
    print("\n - Used context size: ", context_size)
    
    print("\n - Number of voices: ",comb_net.rhythm_net.n_voices)
    
    
    print("\n - Expected rhythm input size: "+
          "(?, ?) with labels in [0, {}]".format(V_rhythm))
    
    print("\n - Expected melody input size: "+
          "(?, ?, {}) with labels in [0, {}]".format(m, V_melody))
    
    
    print("\n - Expected metaData input size: " + 
          "(?, {})".format(meta_len))
    
    print("\n", "-"*40)
    #%%
    
    batch_size = 5
    bar_length = 4
    
    example_rhythm_contexts = [rand.randint(0, V_rhythm, size=(batch_size, bar_length))
                                    for _ in range(context_size)]
    
    
    example_melody_contexts = rand.randint(0, V_melody, size=(batch_size, context_size, m))
    
    example_metaData = rand.random(size=(batch_size, meta_len))
    
    
    #%%
    
    # asterisk on example_rhythm_contexts is important
    example_output = comb_net.predict(x=[*example_rhythm_contexts, 
                                         example_melody_contexts,
                                         example_metaData])
    
    
    sampled_rhythm = np.argmax(example_output[0], axis=-1)
    sampled_melody = np.argmax(example_output[1], axis=-1)
    
    
    
