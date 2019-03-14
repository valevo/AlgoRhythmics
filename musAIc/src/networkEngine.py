import multiprocessing
import threading
import random
import numpy as np
import numpy.random as rand
import pickle as pkl
from fractions import Fraction

from utils import *

from v4.Nets.CombinedNetwork import CombinedNetwork
#from keras.models import load_model

class Network():
    def __init__(self, name):
        self.name = name

    def generate_bar(self, **kwargs):
        ''' Requests the next bar '''
        pass


class PlayerV4():
    ''' First generation of Neural Net Player '''
    def __init__(self, _id, ins_panel=None):
        self._id = _id
        self.ins_panel = ins_panel
        if ins_panel:
            self.metaParameters = self.update_params(ins_panel.getMetaParams())
            print('Found instrument panel, using paramters from knobs')
        else:
            self.metaParameters = TEST_MD
            print('Using default TEST metadata')

        print('Loading Player', _id)

        with open('./v4/Nets/rhythmDict.pkl', 'rb') as f:
            self.rhythmDict = pkl.load(f)

        self.indexDict = {v: k for k, v in self.rhythmDict.items()}

        weights_folder = "v4/Nets/weights/Wed_Mar_13_23-08-08_2019/"
        self.comb_net = CombinedNetwork.from_saved_custom(weights_folder,
                                                     generation=True,
                                                     compile_now=False)

        self.context_size = self.comb_net.params["context_size"]

        self.V_rhythm = self.comb_net.params["bar_embed_params"][0]
        self.m, self.V_melody = self.comb_net.params["melody_net_params"][0], self.comb_net.params["melody_net_params"][1]
        meta_len = comb_net.params["meta_len"]
        #self.meta_len = len(self.metaParameters)

        print("-"*40,  "\nINFO FOR LOADED NET:", self.comb_net)
        print(" - Used context size: ", self.context_size)
        print(" - Number of voices: ",self.comb_net.rhythm_net.n_voices)
        print(" - Expected rhythm input size: "+
              "(?, ?) with labels in [0, {}]".format(self.V_rhythm))
        print(" - Expected melody input size: "+
              "(?, ?, {}) with labels in [0, {}]".format(self.m, self.V_melody))
        print(" - Expected metaData input size: " +
              "(?, {})".format(self.meta_len))
        print("-"*40)

        self.batch_size = 1
        self.bar_length = 4

        self.rhythm_contexts = [ 773*np.ones((self.batch_size, self.bar_length)) for _ in range(self.context_size)]
        #self.rhythm_contexts = [rand.randint(0, self.V_rhythm, size=(self.batch_size, self.bar_length))
        #                                for _ in range(self.context_size)]

        self.melody_contexts = rand.randint(1, 13, size=(self.batch_size, self.context_size, self.m))

        self.prepare_meta_data()

        print('Player {} loaded\n'.format(self._id))

    def generate_bar(self, **kwargs):
        # asterisk on example_rhythm_contexts is important
        # predict...
        output = self.comb_net.predict(x=[*self.rhythm_contexts,
                                             self.melody_contexts,
                                             self.metaData])

        # get rhythm and melody...
        sampled_rhythm = np.argmax(output[0], axis=-1)
        sampled_melody = np.argmax(output[1], axis=-1)

        print('sampled rhythm...', sampled_rhythm)
        print('sampled melody...', sampled_melody)

        # update history...
        self.rhythm_contexts.append(sampled_rhythm)
        self.rhythm_contexts = self.rhythm_contexts[1:]

        self.melody_contexts = np.append(self.melody_contexts, [sampled_melody], axis=1)[:, 1:, :]

        # convert to bar...
        rhythm = []
        for b in sampled_rhythm[0]:
            rhythm.append(self.indexDict[b])

        melody = [int(n) for n in sampled_melody[0]]
        octaves =[3]*48
        #octaves =[int(x) for x in rand.choice([2, 3, 4], 48, p=[0.2, 0.7, 0.1])]

        bar = parseBarData(melody, octaves, rhythm)

        return bar

    def update_params(self, params):
        for k in params.keys():
            self.metaParameters[k] = params[k]
        print('Paramters for', self._id, 'updated')
        self.prepare_meta_data()

    def update_contexts(self, stream, updateMeta=False):
        ''' update the contexts based on the given stream, i.e. after recording '''
        stream_data = convertStreamToData(stream)
        new_rhythm_context = []
        new_melody_context = np.zeros(shape=(1, self.context_size, 48))
        for i in range(self.context_size):
            if self.context_size-i > len(stream):
                #new_melody_context[0, i, :] = [0]*48
                new_rhythm_context.append(np.array([[self.rhythmDict[(0.0,)]]*4]))
            else:
                new_melody_context[0, i, :] = stream_data['melody']['notes'][-self.context_size+i]
                r_bar = stream_data['rhythm'][-self.context_size+i]
                bar = np.array([[self.rhythmDict[b] for b in r_bar]])
                new_rhythm_context.append(bar)

        self.melody_contexts = new_melody_context
        self.rhythm_contexts = new_rhythm_context

        if updateMeta:
            self.update_params(stream.getMetaAnalysis())
            if self.ins_panel:
                self.ins_panel.updateMetaParams(self.metaParameters)

    def prepare_meta_data(self):
        values = []
        for k in sorted(self.metaParameters.keys()):
            if k == 'ts':
                frac = Fraction(self.metaParameters[k], _normalize=False)
                values.extend([frac.numerator, frac.denominator])
            else:
                assert isinstance(self.metaParameters[k], (float, int))
                values.append(self.metaParameters[k])

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
                #md = {'ts': '4/4', 'span': 10, 'jump': 1.511111111111111, 'cDens': 0.2391304347826087, 'cDepth': 0.0, 'tCent': 62.97826086956522, 'rDens': 1.0681818181818181, 'expression': 0}
                if len(req) > 3:
                    self.models[_id] = PlayerV4(_id, req[3])
                else:
                    self.models[_id] = PlayerV4(_id)

            elif req[0] == 1:
                # instument requests new bar
                _id = req[1]
                bar = self.models[_id].generate_bar(confidence=req[2])
                #bar = self.create_bar(_id, req[2])
                self.return_queues[_id].put({'bar': bar})

            elif req[0] == 2:
                # instrument wants to regenerate data (e.g. after recording)
                _id = req[1]
                stream = req[2]
                self.models[_id].update_contexts(stream)

            elif req[0] == 3:
                # instrument has updated it's parameters
                _id = req[1]
                self.models[_id].update_params(req[2])

            elif req[0] == 4:
                # get metaData parameters
                _id = req[1]
                md = self.models[_id].metaParameters
                self.return_queues[_id].put({'md': md})

            elif req[0] == -1:
                # remove the model and data
                _id = req[1]
                del self.models[_id]
                #del self.data[_id]
                del self.return_queues[_id]


TEST_MD = {'ts': '4/4', 'span': 10, 'jump': 1.511111111111111, 'cDens': 0.2391304347826087, 'cDepth': 0.0, 'tCent': 62.97826086956522, 'rDens': 1.0681818181818181, 'expression': 0}
