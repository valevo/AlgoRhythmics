import time
import multiprocessing
import threading
import random
import numpy as np
import numpy.random as rand
import pickle as pkl
from fractions import Fraction
from copy import deepcopy

from utils import *

from player9 import NNPlayer9
from player9C import NNPlayer9C
from player10 import NNPlayer10
from player10C import NNPlayer10C

# =====================================
# either 9 or 10, +.5 for chord version
PLAYER_VERSION = 9.5
# =====================================

class Player():
    def __init__(self, _id):
        self._id = _id

    def generate_bar(self, **kwargs):
        ''' Requests the next bar '''
        return None

    def get_contexts(self):
        return None

    def update_params(self, params):
        print('Params updated:')
        print(params)

class BasicPlayer(Player):
    '''
    Super basic player network, does not coroporate with other musicians or use
    additional information. FOR TESTING ONLY
    '''
    def __init__(self, _id):
        super().__init__(_id)

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

        return bar, None

    def update_params(self, params):
        print('Params updated:')
        print(params)


class DataReader(Player):
    ''' Player that reads in musical data of the Melody, Rhythm, Chords format '''
    def __init__(self, _id):
        super().__init__(_id)

        self.notes      = None 
        self.octaves    = None 
        self.rhythm     = None 

        self.rhythm_contexts = None
        self.melody_contexts = None

        self.current_bar = 0

        #self.open_data('./testData.pkl')

        print('DataReader initialised')

    def generate_bar(self, **kwargs):
        ''' Reads the next bar from the file '''

        if 'load_file' in kwargs:
            if kwargs['load_file'] and kwargs['load_file'] != 'None':
                print('Loading file')
                self.open_data(kwargs['load_file'])

        if not self.notes:
            return None, None

        try:
            bar = parseBarData(self.notes[self.current_bar],
                                self.octaves[self.current_bar],
                                self.rhythm[self.current_bar])
            context = (self.rhythm[self.current_bar], np.array([self.notes[self.current_bar]]))
        except IndexError:
            bar = None
            context = None

        self.current_bar += 1

        return bar, context

    def get_contexts(self):
        return (self.rhythm_contexts, self.melody_contexts)
    
    def open_data(self, dir):
        #'./testData.pkl'
        print('open_data', dir)
        try:
            with open(dir, 'rb') as f:
                self.music_data = pkl.load(f)
        except:
            print('Could not find {}...'.format(dir))
            return

        self.current_bar = 0
        self.notes      = self.music_data[0]['melody']['notes']
        self.octaves    = self.music_data[0]['melody']['octaves']
        self.rhythm     = self.music_data[0]['rhythm']

        self.rhythm_contexts = np.array(self.rhythm)
        self.melody_contexts = np.array([self.notes])

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

            #print(req)

            _id = req[1]

            if req[0] == 0:
                # new instrument id
                self.return_queues[_id] = req[2]
                #model = BasicPlayer(_id)

                if req[3] == 0:
                    # load network
                    print('Loading Player version', PLAYER_VERSION)
                    if PLAYER_VERSION == 9:
                        model = NNPlayer9(_id)
                    elif PLAYER_VERSION == 9.5:
                        model = NNPlayer9C(_id)
                    elif PLAYER_VERSION == 10:
                        model = NNPlayer10(_id)
                    elif PLAYER_VERSION == 10.5:
                        model = NNPlayer10C(_id)
                    else:
                        print('Loading DataReader.')
                        model = DataReader(_id)
                else:
                    # load DataReader
                    model = DataReader(_id)

                self.models[_id] = model

                contexts = (model.rhythm_contexts, model.melody_contexts)
                try:
                    rd = model.rhythmDict
                except AttributeError:
                    rd = None

                self.return_queues[_id].put({'init_contexts': contexts, 'rhythm_dict': rd})

            elif req[0] == 1:
                # instument requests new bar
                if isinstance(self.models[_id], DataReader):
                    kwargs = req[2]
                    while True:
                        bar, context = self.models[_id].generate_bar(**kwargs)
                        if not bar:
                            break
                        self.return_queues[_id].put({'bar': bar, 'context': context})
                        kwargs['load_file'] = None
                else:
                    bar, context = self.models[_id].generate_bar(**req[2])
                    self.return_queues[_id].put({'bar': bar, 'context': context})

            elif req[0] == 2:
                # instrument wants to regenerate data (e.g. after recording)
                stream = req[2]
                self.models[_id].update_contexts(stream)

            elif req[0] == 3:
                # instrument has updated it's parameters
                self.models[_id].update_params(req[2])

            elif req[0] == 4:
                # get metaData parameters
                md = self.models[_id].metaParameters
                self.return_queues[_id].put({'md': md})

            elif req[0] == 5:
                # get contexts
                contexts = self.models[_id].getContexts()
                self.return_queues[_id].put({'contexts': contexts})

            elif req[0] == 6:
                # load contexts from file
                if isinstance(self.models[_id], DataReader):
                    self.models[_id].open_data(req[2])

            elif req[0] == -1:
                # remove the model and data
                del self.models[_id]
                del self.return_queues[_id]


TEST_MD = {'ts': '4/4', 'span': 10, 'jump': 1.511111111111111, 'cDens': 0.2391304347826087, 'cDepth': 0.0, 'tCent': 62.97826086956522, 'rDens': 1.0681818181818181, 'pos': 0.5, 'expression': 0}
