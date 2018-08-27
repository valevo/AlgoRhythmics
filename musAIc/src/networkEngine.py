import multiprocessing
import threading
import random
import numpy as np
import pickle as pkl

from keras.models import load_model

class Network(multiprocessing.Process):
    def __init__(self, request_queue):
        super(Network, self).__init__()

        self.request_queue = request_queue
        self.return_queues = {}
        self.data = {}
        self.models = {}

        with open('./note_dict.pkl', 'rb') as f:
            self.word_to_id = pkl.load(f)

        self.vocab = len(self.word_to_id)
        self.id_to_word = dict(zip(self.word_to_id.values(), self.word_to_id.keys()))

        self.stop_request = threading.Event()

        print('Initiated network manager')


    def run(self):
        print('Network started')
        while not self.stop_request.isSet():
            # first check if any requests
            try:
                req = self.request_queue.get(timeout=1)
            except:
                continue

            if req[0] == 0:
                # new instrument id
                _id = req[1]
                self.return_queues[_id] = req[2]
                self.new_model(_id)

            elif req[0] == 1:
                # instument requests new bar
                _id = req[1]
                bar = self.create_bar(_id, req[2])
                self.return_queues[_id].put(bar)

            elif req[0] == -1:
                # remove the model and data
                _id = req[1]
                del self.models[_id]
                del self.data[_id]
                del self.return_queues[_id]


    def new_model(self, _id):
        model = load_model('./final_model.hdf5')
        self.models[_id] = model
        data = []
        for _ in range(30):
            data.append(random.randint(0, self.vocab-1))
        data = np.array([data])
        self.data[_id] = data
        print('Model {} loaded and primed'.format(_id))

    def create_bar(self, _id, confidence=0):
        bar = {}
        t = 0
        model = self.models[_id]
        data = self.data[_id]
        tries = 0

        while t < 4:
            prediction = model.predict(data)
            idxs = np.argsort(-prediction[:, 29, :])[0]
            predict_word = idxs[0 + confidence]
            word = self.id_to_word[predict_word]
            if word == 'EOP':
                # pick next best...
                predict_word = idxs[1 + confidence]
                word = self.id_to_word[predict_word]

            bar[t] = word[0]
            t += word[1]
            data = np.append(data[0][1:], predict_word).reshape((1, 30))

        self.data[_id] = data

        return bar
