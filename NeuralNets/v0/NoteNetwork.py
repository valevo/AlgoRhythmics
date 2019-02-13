# -*- coding: utf-8 -*-


from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from keras.metrics import categorical_accuracy

class NoteNetwork(Model):
    def __init__(self, num_categories, embed_size, lstm_size, compile_now=True):
        embedding_layer = Embedding(input_dim=num_categories,
                             output_dim=embed_size)
        lstm_layer = LSTM(lstm_size, return_sequences=True)
        lstm_layer_prev = LSTM(lstm_size, return_state=True)
        output_layer = TimeDistributed(Dense(num_categories, 
                                             activation="softmax"))
        
        
        inputs = Input(shape=(None,))
        prev_bar = Input(shape=(None,))
        
        embedded = embedding_layer(inputs)
        prev_bar_embedded = embedding_layer(prev_bar)
        
        prev_bar_output, prev_bar_h_state, prev_bar_cell_state = \
                                            lstm_layer_prev(prev_bar_embedded)
        lstm_outputs = lstm_layer(embedded, 
                                  initial_state=(prev_bar_h_state, 
                                                 prev_bar_cell_state))
        
        preds = output_layer(lstm_outputs)
        
        super().__init__(inputs=[prev_bar, inputs], outputs=preds)
        
        if compile_now:
            self.compile_default()
        
        
    def compile_default(self):
        self.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[categorical_accuracy])
        
    @classmethod    
    def from_weights(cls, filename, **kwargs):
        m = cls(**kwargs)
        m.load_weights(filename)
        return m