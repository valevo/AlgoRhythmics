# -*- coding: utf-8 -*-

#%%

import keras.backend as K
from keras.layers import Input, Embedding,\
             Bidirectional, LSTM, Dense, TimeDistributed, Lambda,\
             RepeatVector             
from keras.models import Model
from keras.objectives import categorical_crossentropy

from keras.utils import plot_model, to_categorical

import numpy as np
import numpy.random as rand

#%%

class GaussVAE(Model):
    def __init__(self, input_len, embed_size, epsilon_std, compile_now=False): 
        self.embed_size = embed_size
        self.epsilon_std = epsilon_std
        
        inputs = Input(shape=(input_len, ))
        
        # encode
        h_enc = Dense(embed_size, activation="tanh")(inputs)
        self.z_mean = Dense(embed_size, name="z_mean")(h_enc)
        self.z_log_var = Dense(embed_size, name="z_var")(h_enc)
        
        # sample
        z = Lambda(self.sample, output_shape=(embed_size, ))([self.z_mean, 
                                                              self.z_log_var])
    
        # decode
        h_dec = Dense(input_len, activation="tanh")(z)
        x_mean = Dense(input_len)(h_dec)    
        x_log_var = Dense(input_len)(h_dec)
    
#        pred = Dense(input_len, activation="linear", name="x_hat")(z)
        
        super().__init__(inputs=inputs, outputs=[x_mean, x_log_var])
        
        if compile_now:
            self.compile_default()
            
            
    def compile_default(self):
        self.compile("adam", 
                     loss=self.vae_loss,
                     metrics=["mean_absolute_error"])
            
    def sample(self, args):
        z_mean_, z_log_var_ = args
        batch_size = K.shape(z_mean_)[0]
        epsilon = K.random_normal(shape=(batch_size, self.embed_size), 
                                  mean=0., stddev=self.epsilon_std)
        return z_mean_ + K.exp(z_log_var_/2) * epsilon
        
        
    def KL_div(self):
        return -0.5*K.sum(1 + 
                          self.z_log_var - 
                          K.square(self.z_mean) - 
                          K.exp(self.z_log_var), axis=-1)

    def gauss_pdf(self, x, x_pred):
        mean, log_var = x_pred
        var = K.exp(log_var)
        pi_cons = K.constant(np.pi)
        Z = K.pow(2*pi_cons*var, 0.5)
        prob = K.log(K.exp(-0.5*K.pow(x - mean, 2) / var) / Z)  
        return prob
        
    
    def vae_loss(self, x, x_pred):
        xent = self.gauss_pdf(x, x_pred)
        KL = self.KL_div()
#        KL_repeated = K.repeat_elements(K.reshape(KL, shape=(K.shape(KL)[0], 1)),
#                                        rep=self.m, axis=-1)
        return xent + KL


#%%

vae = GaussVAE(input_len=10, embed_size=8, epsilon_std=1.0, compile_now=True)


#%%

xs = rand.random(size=(100, 10))

ys = xs[:, :]


#%%

vae.fit(x=xs, y=ys, epochs=1)


#%%
class RecurrentGaussVAE(Model):
    def __init__(self, m, num_categories, embed_size, 
                 enc_lstm_size, latent_dim, dec_lstm_size,
                 epsilon_std, compile_now=True):
        
        self.m = m
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std
        
        
        self.inputs = Input(shape=(m,))
#        cur_seq_len = K.shape(inputs)[1]
        
        embedded = Embedding(num_categories, 
                            embed_size,
                            input_length=m)(self.inputs)

        h1 = Bidirectional(LSTM(enc_lstm_size, 
                        return_sequences=False, 
                        name='encoder_lstm'), merge_mode='concat')(embedded)

        self.z_mean = Dense(self.latent_dim, name='z_mean', activation='linear')(h1)
        self.z_log_var = Dense(self.latent_dim, name='z_var', activation='linear')(h1)

        encoded = Lambda(self.sampling, 
                 output_shape=(self.latent_dim,), 
                 name='lambda')([self.z_mean, self.z_log_var])

        repeated_context = RepeatVector(m)(encoded)

        decoded = LSTM(dec_lstm_size, 
               return_sequences=True, name='dec_lstm')(repeated_context)

        self.preds = TimeDistributed(Dense(num_categories, activation='softmax'), 
                          name='softmax_layer')(decoded)
        
        
        super().__init__(inputs=self.inputs, outputs=self.preds)
        
        if compile_now:
            self.compile_default()
    
    
    def KL_div(self):
        return -0.5*K.sum(1 + 
                          self.z_log_var - 
                          K.square(self.z_mean) - 
                          K.exp(self.z_log_var), axis=-1)
    
    def vae_loss(self, x, x_pred):
        xent = categorical_crossentropy(x, x_pred)
#        cur_seq_len = K.shape(x)[1]
        KL = self.KL_div()
        KL_repeated = K.repeat_elements(K.reshape(KL, shape=(K.shape(KL)[0], 1)),
                                        rep=self.m, axis=-1)
        return xent + KL_repeated

    def sampling(self, args):
        z_mean_, z_log_var_ = args
        batch_size = K.shape(z_mean_)[0]
        epsilon = K.random_normal(shape=(batch_size, self.latent_dim), 
                                  mean=0., stddev=self.epsilon_std)
        return z_mean_ + K.exp(z_log_var_/2) * epsilon
    

    
    def compile_default(self):
        self.compile(optimizer='Adam',
                                 loss=self.vae_loss,
                                 metrics=["categorical_accuracy"])
        
        
    def get_encoder(self):
        return Model(inputs=self.inputs, outputs=[self.preds, 
                                                  self.z_mean, 
                                                  self.z_log_var])
    
    def sampling_numpy(self, some_means, some_stds):
        assert some_means.shape == some_stds.shape
    
        eps = rand.normal(loc=0., 
                          scale=self.epsilon_std**2, 
                          size=some_means.shape)
        return some_means + some_stds * eps
    
    def get_decoder(self):
        return Model(inputs=[self.z_mean, self.z_log_var],
                     outputs=self.preds)


#%%

n_cat = 4
m=3

embed_size = 8
enc_size=12
l_dim=3
dec_size = 16

epsilon_std = .01



vae = RecurrentGaussVAE(m, n_cat, 
                        embed_size, enc_size, l_dim, dec_size, 
                        epsilon_std)



#%%
xs = rand.choice(n_cat, size=(1000, m))

xs_cat = to_categorical(xs)
#%%

vae.fit(xs, xs_cat, epochs=50, verbose=0)


#%%

vae.fit(xs, xs_cat, epochs=3, verbose=2)


test_n = 5
test_start = rand.randint(0, len(xs)-test_n)

gen = vae.get_encoder()
pred_x, pred_mean, pred_log_var = gen.predict(xs[test_start:test_start+test_n])
pred_std = np.exp(pred_log_var/2)

print(xs[test_start:test_n+test_start], np.argmax(pred_x, axis=-1))
print(pred_mean)
print(pred_std)

print(vae.sampling_numpy(pred_mean, pred_std))


#%%
