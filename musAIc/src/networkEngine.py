import multiprocessing
import threading
import random
import numpy as np
import numpy.random as rand
import pickle as pkl
from fractions import Fraction

from utils import *

from v3.Nets.CombinedNetwork import CombinedNetwork
#from keras.models import load_model

class Network():
    def __init__(self, name):
        self.name = name

    def generate_bar(self, **kwargs):
        ''' Requests the next bar '''
        pass


class PlayerV3():
    ''' First generation of Neural Net Player '''
    def __init__(self, _id, metaParameters):
        self._id = _id
        self.metaParameters = metaParameters

        print('Loading net', _id)
        with open('/home/arran/AlgoRhythmics/musAIc/src/v3/Nets/rhythmDict.pkl', 'rb') as f:
            self.rhythmDict = pkl.load(f)

        weights_folder = "v3/Nets/weights/Thu_Mar__7_17-28-16_2019/"
        self.comb_net = CombinedNetwork.from_saved_custom(weights_folder,
                                                     generation=True,
                                                     compile_now=False)

        self.context_size = self.comb_net.params["context_size"]

        self.V_rhythm = self.comb_net.params["bar_embed_params"][0]
        self.m, self.V_melody = self.comb_net.params["melody_net_params"][0], self.comb_net.params["melody_net_params"][1]
        #meta_len = comb_net.params["meta_len"]
        self.meta_len = len(self.metaParameters)

        print("\n", "-"*40,  "\nINFO FOR LOADED NET:", self.comb_net)
        print("\n - Used context size: ", self.context_size)
        print("\n - Number of voices: ",self.comb_net.rhythm_net.n_voices)
        print("\n - Expected rhythm input size: "+
              "(?, ?) with labels in [0, {}]".format(self.V_rhythm))
        print("\n - Expected melody input size: "+
              "(?, ?, {}) with labels in [0, {}]".format(self.m, self.V_melody))
        print("\n - Expected metaData input size: " +
              "(?, {})".format(self.meta_len))
        print("\n", "-"*40)

        self.batch_size = 1
        self.bar_length = 4

        self.rhythm_contexts = [rand.randint(0, self.V_rhythm, size=(self.batch_size, self.bar_length))
                                        for _ in range(self.context_size)]
        self.melody_contexts = rand.randint(0, self.V_melody, size=(self.batch_size, self.context_size, self.m))
        #example_metaData = rand.random(size=(batch_size, meta_len))
        self.prepare_meta_data()

        #self.metaData = np.tile(mData, (self.batch_size, 1))


    def generate_bar(self, **kwargs):
        # asterisk on example_rhythm_contexts is important
        # predict...
        output = self.comb_net.predict(x=[*self.rhythm_contexts,
                                             self.melody_contexts,
                                             self.metaData])

        # get rhythm and melody...
        sampled_rhythm = np.argmax(output[0], axis=-1)
        sampled_melody = np.argmax(output[1], axis=-1)

        # update history...
        self.rhythm_contexts.append(sampled_rhythm)
        self.rhythm_contexts = self.rhythm_contexts[1:]

        self.melody_contexts = np.append(self.melody_contexts, [sampled_melody], axis=1)[:, 1:, :]

        # convert to bar...
        rhythm = []
        for b in sampled_rhythm[0]:
            rhythm.append(self.rhythmDict[b])

        melody = [int(n) for n in sampled_melody[0]]
        octaves =[int(x) for x in rand.randint(3, 5, 48)]
        #print('Melody:', melody)
        #print('rhythm:', rhythm)

        bar = parseBarData(melody, octaves, rhythm)
        #print('Bar generated', bar)

        return bar

    def update_params(self, params):
        for k in params.keys():
            self.metaParameters[k] = params[k]
        print('Paramters for', self._id, 'updated')
        self.prepare_meta_data()

    def prepare_meta_data(self):
        values = []
        for k in sorted(self.metaParameters.keys()):
            if k == 'ts':
                frac = Fraction(self.metaParameters[k], _normalize=False)
                values.extend([frac.numerator, frac.denominator])
            else:
                assert isinstance(self.metaParameters[k], (float, int))
                values.append(self.metaParameters[k])

        print(values)
        self.metaData = np.tile(values, (self.batch_size, 1))


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

        print('DataReader initialised')

    def generate_bar(self, **kwargs):
        ''' Reads the next bar from the file '''

        try:
            bar = parseBarData(self.notes[self.current_bar],
                                self.octaves[self.current_bar],
                                self.rhythm[self.current_bar])
        except IndexError:
            bar = None

        self.current_bar += 1

        print('Bar generated', bar)
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
                #self.models[_id] = DataReader(_id)
                #self.models[_id] = BasicPlayer(_id)
                md = {'ts': '4/4', 'span': 10, 'jump': 1.511111111111111, 'cDens': 0.2391304347826087, 'cDepth': 0.0, 'tCent': 62.97826086956522, 'rDens': 1.0681818181818181, 'expression': 0}
                self.models[_id] = PlayerV3(_id, md)

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


