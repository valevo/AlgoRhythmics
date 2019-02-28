import multiprocessing
import threading
import random
import numpy as np
import pickle as pkl

from utils import *

#from keras.models import load_model

class Network():
    def __init__(self, name):
        self.name = name

    def generate_bar(self, **kwargs):
        ''' Requests the next bar '''
        pass


class BasicPlayer():
    '''
    Super basic player network, does not coroporate with other musicians or use
    additional information. FOR TESTING ONLY
    '''
    def __init__(self, _id):
        self.id = _id

        # load model...
        #self.model = load_model('./final_model.hdf5')

        with open('./note_dict.pkl', 'rb') as f:
            self.word_to_id = pkl.load(f)

        self.vocab = len(self.word_to_id)
        self.id_to_word = dict(zip(self.word_to_id.values(), self.word_to_id.keys()))

        # initialise random data...
        self.data = []
        for _ in range(30):
            self.data.append(random.randint(0, self.vocab-1))
        self.data = np.array([self.data])

        print('Model {} loaded and primed'.format(self.id))


    def generate_bar(self, **kwargs):
        bar = {}
        t = 0

        try:
            confidence = kwargs['confidence']
        except:
            confidence = 0

        while t < 4:
            prediction = self.model.predict(self.data)
            idxs = np.argsort(-prediction[:, 29, :])[0]
            predict_word = idxs[0 + confidence]
            word = self.id_to_word[predict_word]
            if word == 'EOP':
                # pick next best...
                predict_word = idxs[1 + confidence]
                word = self.id_to_word[predict_word]

            bar[t] = word[0]
            t += word[1]
            self.data = np.append(self.data[0][1:], predict_word).reshape((1, 30))

        return bar

    def update_params(self, params):
        print('Params updated:')
        print(params)


class DataReader():
    ''' Player that reads in musical data of the Melody, Rhythm, Chords format '''
    def __init__(self, _id):
        with open('./testData.pkl', 'rb') as f:
            self.music_data = pkl.load(f)

        self._id = _id
        self.current_bar = 0

        self.notes      = self.music_data['instruments'][_id]['melody']['notes']
        self.octaves    = self.music_data['instruments'][_id]['melody']['octaves']
        self.rhythm     = self.music_data['instruments'][_id]['rhythm']

    def generate_bar(self, **kwargs):
        ''' Reads the next bar from the file '''

        bar = parseBarData(self.notes[self.current_bar],
                            self.octaves[self.current_bar],
                            self.rhythm[self.current_bar])

        self.current_bar += 1

        return bar


    def update_params(self, params):
        print('Params updated [{}]:'.format(self._id))
        print(params)



class NetworkManager(multiprocessing.Process):
    def __init__(self, request_queue):
        super(NetworkManager, self).__init__()

        self.request_queue = request_queue
        self.return_queues = {}
        #self.data = {}
        self.models = {}

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
                self.models[_id] = DataReader(_id)
                #self.models[_id] = BasicPlayer(_id)

            elif req[0] == 1:
                # instument requests new bar
                _id = req[1]
                bar = self.models[_id].generate_bar(confidence=req[2])
                #bar = self.create_bar(_id, req[2])
                self.return_queues[_id].put(bar)

            elif req[0] == 2:
                # instrument wants to regenerate data (e.g. after recording)
                _id = req[1]

                pass

            elif req[0] == 3:
                # instrument has updated it's parameters
                _id = req[1]
                self.models[_id].update_params(req[2])

            elif req[0] == -1:
                # remove the model and data
                _id = req[1]
                del self.models[_id]
                #del self.data[_id]
                del self.return_queues[_id]


