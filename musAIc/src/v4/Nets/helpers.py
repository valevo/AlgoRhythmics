# -*- coding: utf-8 -*-


from keras.layers import Input, Lambda, Concatenate, RepeatVector
from keras.models import Model, load_model
from keras.utils import to_categorical, plot_model
import keras.backend as K

import tensorflow as tf

import numpy as np
import numpy.random as rand



class ConcatAndRepeat(Lambda):
    def __init__(self, tensor_list, n_repeats, concat_axis=-1, **kwargs):
        
        concated = Concatenate(axis=concat_axis)(tensor_list)
        
        super(MyConcat, self).__init__()
        
        
    def _repeat(self, args):
        some_bar, vec = args
        bar_len = K.shape(some_bar)[-1]
        return RepeatVector(bar_len)(vec)