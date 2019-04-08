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

#from v5.Nets.CombinedNetwork import CombinedNetwork
from v9_dev.Nets.CombinedNetwork import CombinedNetwork
from v9_dev.Nets.MetaEmbedding import MetaEmbedding
from v9_dev.Nets.MetaPredictor import MetaPredictor

class Player():
    def __init__(self, _id):
        self._id = _id

    def generate_bar(self, **kwargs):
        ''' Requests the next bar '''
        return None

    def get_contexts(self):
        return None

class NNPlayer9(Player):
    ''' Neural Net Player (version 9)'''
    def __init__(self, _id, ins_panel=None):
        super().__init__(_id)
        self.ins_panel = ins_panel
        if ins_panel:
            self.metaParameters = self.update_params(ins_panel.getMetaParams())
            print('Found instrument panel, using paramters from knobs')
        else:
            self.metaParameters = deepcopy(TEST_MD)
            print('Using default TEST metadata')

        print('Loading Player', _id)

        with open('./v9_dev/Trainings/first_with_lead/DataGenerator.conversion_params', 'rb') as f:
            conversion_params = pkl.load(f)
            self.rhythmDict = conversion_params['rhythm']

        self.indexDict = {v: k for k, v in self.rhythmDict.items()}

        self.metaEmbedder = MetaEmbedding.from_saved_custom('./v9_dev/Trainings/first_with_lead/meta/')
        metaPredictor = MetaPredictor.from_saved_custom('./v9_dev/Trainings/first_with_lead/meta/')

        weights_folder = "./v9_dev/Trainings/first_with_lead/weights/_checkpoint_19/"
        self.comb_net = CombinedNetwork.from_saved_custom(weights_folder, metaPredictor,
                                                     generation=True,
                                                     compile_now=False)

        print('Repr:', repr(self.comb_net))

        self.context_size = self.comb_net.params["context_size"]

        #print(self.comb_net.params)
        self.V_rhythm = self.comb_net.params["rhythm_net_params"][2]
        self.m = self.comb_net.params["melody_bar_len"]
        self.V_melody = self.comb_net.params["melody_net_params"][3]

        self.metaEmbedSize = self.metaEmbedder.embed_size

        self.batch_size = 1
        self.bar_length = 4

        if _id == 0:
            self.inject_params = ({'qb', 'eb'}, 'maj')
        else:
            self.inject_params = ({'lb', 'tb'}, 'maj')

        self.inject_contexts()
        self.prepare_meta_data()

        self.outputFile = './rhythm_dist_{}.csv'.format(_id)
        with open(self.outputFile, 'w') as f:
            pass

        print('Player {} loaded\n'.format(_id))


    def inject_contexts(self):
        r_selection = self.inject_params[0]
        scale = self.inject_params[1]

        holdBeat =    [self.rhythmDict[()]]
        quarterBeat = [self.rhythmDict[(0.0,)]]
        eighthBeats = [self.rhythmDict[(0.0, 0.5)],
                       self.rhythmDict[(0.5,)]]
        fastBeat =    [self.rhythmDict[0.0, 0.25, 0.5, 0.75]]
        triplets =    [self.rhythmDict[(0.0, 0.3333, 0.6667)],
                       self.rhythmDict[(0.0, 0.6667)]]

        print('Contexts with: ', r_selection, scale)

        rhythm_pool = []

        if 'qb' in r_selection:
            rhythm_pool.extend(quarterBeat)

        if 'eb' in r_selection:
            rhythm_pool.extend(eighthBeats)

        if 'fb' in r_selection:
            rhythm_pool.extend(fastBeat)

        if 'tb' in r_selection:
            rhythm_pool.extend(triplets)

        if 'lb' in r_selection:
            rhythm_pool.extend(holdBeat)

        if len(rhythm_pool) == 0:
            rhythm_pool = quarterBeat

        self.rhythm_contexts = [rand.choice(rhythm_pool, size=(self.batch_size, self.bar_length))
                                for _ in range(self.context_size)]

        if scale == 'maj':
            self.melody_contexts = rand.choice([1, 3, 5, 6, 8, 10, 12],
                                               size=(self.batch_size, self.context_size, self.m))
        elif scale == 'min':
            self.melody_contexts = rand.choice([1, 3, 4, 6, 8, 10, 11],
                                               size=(self.batch_size, self.context_size, self.m))
        elif scale == 'pen':
            self.melody_contexts = rand.choice([1, 4, 6, 8, 11],
                                               size=(self.batch_size, self.context_size, self.m))
        elif scale == '5th':
            self.melody_contexts = rand.choice([1, 8],
                                               size=(self.batch_size, self.context_size, self.m))


    def generate_bar(self, **kwargs):
        ''' Generate a bar. Possible kwargs are:
            - 'lead_bar': tuple of lead's contexts
            - 'loop_rhythm': number of bars to repeat
            - 'lm': lead mode
            - 'sm': sample mode
            - 'um': context mode
            - 'chord': include chords of depth
            - 'injection_params': parameters for injection
        '''

        #print(kwargs)

        # -- LEAD
        if 'lm' in kwargs:
            lm = kwargs['lm']
        else:
            lm = 2

        lead_mode = {0: 'none',
                     1: 'both',
                     2: 'melody'}[lm]

        if 'lead_bar' in kwargs:
            if not kwargs['lead_bar']:
                lead_mode = 'none'
        else:
            lead_mode = 'none'

        if lead_mode == 'none':
            # lead context is own context
            lead_r_context = self.rhythm_contexts[-1]
            lead_m_context = self.melody_contexts[:, -1:, :]

        elif lead_mode == 'both':
            # follow lead's rhythm and melody
            lead_r_context = kwargs['lead_bar'][0]
            lead_m_context = kwargs['lead_bar'][1]

        elif lead_mode == 'melody':
            # only follow lead's melody
            lead_r_context = self.rhythm_contexts[-1]
            lead_m_context = kwargs['lead_bar'][1]


        output = self.comb_net.predict(x=[*self.rhythm_contexts,
                                             self.melody_contexts,
                                             self.embeddedMetaData,
                                             lead_r_context,
                                             lead_m_context])

        # write output distribution to file for analysis
        with open(self.outputFile, 'a') as f:
            for dist in output[1][0]:
                f.write(', '.join([str(x) for x in dist]) + '\n')


        # -- SAMPLE
        if 'sm' in kwargs:
            sm = kwargs['sm']
        else:
            sm = 1

        sample_mode = {0: 'argmax',
                       1: 'dist',
                       2: 'top_dist'}[sm]

        top_rhythm = np.argmax(output[0], axis=-1)
        top_melody = np.argmax(output[1], axis=-1)

        if 'chord' in kwargs:
            chord_num = kwargs['chord']
        else:
            chord_num = 1

        if sample_mode == 'argmax':
            # Deterministic playback...
            sampled_rhythm = top_rhythm
            sampled_melody = top_melody
            sampled_chords = [list(rand.choice(self.V_melody, p=curr_p, size=chord_num,
                                           replace=True)) for curr_p in output[1][0]]
        elif sample_mode == 'dist':
            # Random playback...
            sampled_rhythm = np.array([[rand.choice(self.V_rhythm, p=curr_p) for curr_p in output[0][0]]])
            sampled_melody = np.array([[rand.choice(self.V_melody, p=curr_p) for curr_p in output[1][0]]])
            sampled_chords = [list(rand.choice(self.V_melody, p=curr_p, size=chord_num,
                                           replace=True)) for curr_p in output[1][0]]
        elif sample_mode == 'top_dist':
            # Random from top 5 predictions....
            r = []
            sampled_chords = []
            for i in range(4):
                top5_rhythm_indices = np.argsort(output[0][0][i], axis=-1)[-5:]

                r_probs = output[0][0][i][top5_rhythm_indices]
                r_probs /= sum(r_probs)

                r.append(rand.choice(top5_rhythm_indices, p=r_probs))
                #print(top5_rhythm_indices, r_probs, r[-1])

            sampled_rhythm = np.array([r])
            #sampled_melody = np.array([[rand.choice(self.V_melody, p=curr_p) for curr_p in output[1][0]]])
            m = []

            for i in range(len(output[1][0][0])):
                top5_m_indices = np.argsort(output[1][0][i], axis=-1)[-5:]
                m_probs = output[1][0][i][top5_m_indices]
                m_probs /= sum(m_probs)

                m.append(rand.choice(top5_m_indices, p=m_probs))
                sampled_chords.append(list(rand.choice(top5_m_indices, p=m_probs, replace=True, size=chord_num)))
            sampled_melody = np.array([m])

        #print('Sampled rhythm:', [self.indexDict[r] for r in sampled_rhythm[0]])

        if 'loopRhythm' in kwargs:
            if kwargs['loopRhythm'] > 0:
                # ignore the sampled rhythm
                print('loopRhythm')
                num = min(len(self.rhythm_contexts), kwargs['loopRhythm'])
                if num > 0:
                    sampled_rhythm = self.rhythm_contexts[-num]

        # -- UPDATE
        if 'um' in kwargs:
            um = kwargs['um']
        else:
            um = 2
        update_mode = {0: 'none',
                       1: 'top',
                       2: 'sampled',
                       3: 'inject'}[um]


        if 'hold' in kwargs:
            if kwargs['hold']:
                update_mode = 'none'

        if update_mode == 'top':
            # use argmax context
            self.rhythm_contexts.append(top_rhythm)
            self.rhythm_contexts = self.rhythm_contexts[1:]
            self.melody_contexts = np.append(self.melody_contexts, [top_melody], axis=1)[:, 1:, :]
        elif update_mode == 'sampled':
            # use sampled
            self.rhythm_contexts.append(sampled_rhythm)
            self.rhythm_contexts = self.rhythm_contexts[1:]
            self.melody_contexts = np.append(self.melody_contexts, [sampled_melody], axis=1)[:, 1:, :]
        elif update_mode == 'inject':
            # inject new bars
            if 'injection_params' in kwargs:
                self.inject_params = kwargs['injection_params']
            self.inject_contexts()
        elif update_mode == 'none':
            # do not update contexts
            pass

        print('lead_mode:', lead_mode, ' sample_mode:', sample_mode, ' update_mode:', update_mode)

        #print(sampled_rhythm)
        #print(sampled_melody)

        # convert to bar...
        rhythm = [self.indexDict[b] for b in sampled_rhythm[0]]
        if chord_num < 2:
            melody = [int(n) for n in sampled_melody[0]]
        else:
            melody = sampled_chords
        octave = [self.metaParameters['tCent']//12 - 1] * 48

        bar = parseBarData(melody, octave, rhythm)

        return bar, (sampled_rhythm, np.array([sampled_melody]))

    def get_contexts(self):
        return (self.rhythm_contexts, self.melody_contexts)

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
        ''' Updates the metaData for the network to use '''
        values = []
        for k in sorted(self.metaParameters.keys()):
            if k == 'ts':
                frac = Fraction(self.metaParameters[k], _normalize=False)
                values.extend([frac.numerator, frac.denominator])
            else:
                assert isinstance(self.metaParameters[k], (float, int))
                values.append(self.metaParameters[k])

        metaData = np.tile(values, (self.batch_size, 1))

        self.embeddedMetaData = self.metaEmbedder.predict(metaData)
        #print(self.embeddedMetaData)
        assert self.embeddedMetaData.ndim == 2 and self.embeddedMetaData.shape[-1] == self.metaEmbedder.embed_size


#class NNPlayer(Player):
#    ''' Neural Net Player '''
#    def __init__(self, _id, ins_panel=None):
#        super().__init__(_id)
#        self.ins_panel = ins_panel
#        if ins_panel:
#            self.metaParameters = self.update_params(ins_panel.getMetaParams())
#            print('Found instrument panel, using paramters from knobs')
#        else:
#            self.metaParameters = TEST_MD
#            print('Using default TEST metadata')
#
#        print('Loading Player', _id)
#
#        with open('./v7/Data/DataGenerator.conversion_params', 'rb') as f:
#            conversion_params = pkl.load(f)
#            # rhythm -> index
#            self.rhythmDict = conversion_params['rhythm']
#
#        # index -> rhythm
#        self.indexDict = {v: k for k, v in self.rhythmDict.items()}
#
#        weights_folder = "./v7/Nets/weights/Tue_Mar_19_23-01-52_2019/"
#        self.comb_net = CombinedNetwork.from_saved_custom(weights_folder,
#                                                     generation=True,
#                                                     compile_now=False)
#
#        self.context_size = self.comb_net.params["context_size"]
#
#        self.V_rhythm = self.comb_net.params["bar_embed_params"][0]
#        self.m, self.V_melody = self.comb_net.params["melody_net_params"][0], self.comb_net.params["melody_net_params"][1]
#        #meta_len = self.comb_net.params["meta_len"]
#        self.meta_len = len(self.metaParameters)
#
#        print("-"*40,  "\nINFO FOR LOADED NET:", self.comb_net)
#        print(" - Used context size: ", self.context_size)
#        print(" - Number of voices: ",self.comb_net.rhythm_net.n_voices)
#        print(" - Expected rhythm input size: "+
#              "(?, ?) with labels in [0, {}]".format(self.V_rhythm))
#        print(" - Expected melody input size: "+
#              "(?, ?, {}) with labels in [0, {}]".format(self.m, self.V_melody))
#        print(" - Expected metaData input size: " +
#              "(?, {})".format(self.meta_len))
#        print("-"*40)
#
#        self.batch_size = 1
#        self.bar_length = 4
#
#        quarterBeat = self.rhythmDict[(0.0,)]
#        self.rhythm_contexts = [ quarterBeat*np.ones((self.batch_size, self.bar_length)) for _ in range(self.context_size)]
#
#        #self.rhythm_contexts = [rand.randint(0, self.V_rhythm, size=(self.batch_size, self.bar_length))
#        #                                for _ in range(self.context_size)]
#
#        #self.melody_contexts = rand.randint(1, 13, size=(self.batch_size, self.context_size, self.m))
#
#        # seed with major scale...
#        self.melody_contexts = rand.choice([1, 3, 5, 6, 8, 10, 12], size=(self.batch_size, self.context_size, self.m))
#
#        # seed with minor scale...
#        #self.melody_contexts = rand.choice([1, 3, 4, 6, 8, 10, 11], size=(self.batch_size, self.context_size, self.m))
#
#        # seed with simple root/fifth...
#        #self.melody_contexts = rand.choice([1, 8], size=(self.batch_size, self.context_size, self.m))
#
#        # Test by loading contexts from the data...
#        #with open('../../Data/music21/music_data_0005.pkl', 'rb') as f:
#        #    data = pkl.load(f)
#
#        #song = data[34]
#        #self.rhythm_contexts = [np.array([[self.rhythmDict[b] for b in m]]) for m in song[0]['rhythm'][:self.context_size]]
#        #self.melody_contexts = np.array([[[0 if n is None else n for n in m] for m in song[0]['melody']['notes'][:self.context_size]]])
#        #self.metaParameters = song[0]['metaData']
#        self.prepare_meta_data()
#
#        print('Player {} loaded\n'.format(self._id))
#
#    def generate_bar(self, **kwargs):
#        # asterisk on example_rhythm_contexts is important
#        # predict...
#
#        lead_r_context = self.rhythm_contexts
#        lead_m_context = self.melody_contexts
#
#        if 'lead_contexts' in kwargs:
#            lc = kwargs['lead_contexts']
#            if lc:
#                lead_r_context = lc[0]
#                lead_m_context = lc[1]
#
#        #print('MetaData for ins {}:'.format(self._id))
#        #print(self.metaData)
#
#        print('Rhythm context:', self.rhythm_contexts)
#
#        output = self.comb_net.predict(x=[*self.rhythm_contexts,
#                                             self.melody_contexts,
#                                             self.metaData])
#
#        # get rhythm and melody...
#        top_rhythm = np.argmax(output[0], axis=-1)
#        top_melody = np.argmax(output[1], axis=-1)
#
#        if False:
#            # Deterministic playback...
#            sampled_rhythm = top_rhythm
#            sampled_melody = top_melody
#        else:
#            # Random playback...
#            sampled_rhythm = np.array([[rand.choice(self.V_rhythm, p=curr_p) for curr_p in output[0][0]]])
#            sampled_melody = np.array([[rand.choice(self.V_melody, p=curr_p) for curr_p in output[1][0]]])
#
#        if 'loopRhythm' in kwargs:
#            # ignore the sampled rhythm
#            num = min(len(self.rhythm_contexts), kwargs['loopRhythm'])
#            if num > 0:
#                sampled_rhythm = self.rhythm_contexts[-num]
#
#        #print('sampled rhythm...', sampled_rhythm)
#        #print('sampled melody...', sampled_melody)
#
#        # update history...
#        if 'hold' in kwargs:
#            if not kwargs['hold']:
#                print('updating context')
#                self.rhythm_contexts.append(top_rhythm)
#                #self.rhythm_contexts.append(sampled_rhythm)
#                self.rhythm_contexts = self.rhythm_contexts[1:]
#        else:
#            print('updating context')
#            self.rhythm_contexts.append(top_rhythm)
#            #self.rhythm_contexts.append(sampled_rhythm)
#            self.rhythm_contexts = self.rhythm_contexts[1:]
#
#
#        # update with top context
#        self.melody_contexts = np.append(self.melody_contexts, [top_melody], axis=1)[:, 1:, :]
#
#        # update with actual context
#        #self.melody_contexts = np.append(self.melody_contexts, [sampled_melody], axis=1)[:, 1:, :]
#
#        # convert to bar...
#        rhythm = [self.indexDict[b] for b in sampled_rhythm[0]]
#        melody = [int(n) for n in sampled_melody[0]]
#        octave = [self.metaParameters['tCent']//12 - 1] * 48
#
#        #octaves =[int(x) for x in rand.choice([2, 3, 4], 48, p=[0.2, 0.7, 0.1])]
#
#        bar = parseBarData(melody, octave, rhythm)
#        #print('Output bar:\n', bar)
#
#        return bar, (self.rhythm_contexts[-1], self.melody_contexts[0, -1, :])
#
#    def get_contexts(self):
#        return (self.rhythm_contexts, self.melody_contexts)
#
#    def update_params(self, params):
#        for k in params.keys():
#            self.metaParameters[k] = params[k]
#        #print('Paramters for', self._id, 'updated')
#        self.prepare_meta_data()
#
#    def update_contexts(self, stream, updateMeta=False):
#        ''' update the contexts based on the given stream, i.e. after recording '''
#        stream_data = convertStreamToData(stream)
#        new_rhythm_context = []
#        new_melody_context = np.zeros(shape=(1, self.context_size, 48))
#        for i in range(self.context_size):
#            if self.context_size-i > len(stream):
#                #new_melody_context[0, i, :] = [0]*48
#                new_rhythm_context.append(np.array([[self.rhythmDict[(0.0,)]]*4]))
#            else:
#                new_melody_context[0, i, :] = stream_data['melody']['notes'][-self.context_size+i]
#                r_bar = stream_data['rhythm'][-self.context_size+i]
#                bar = np.array([[self.rhythmDict[b] for b in r_bar]])
#                new_rhythm_context.append(bar)
#
#        self.melody_contexts = new_melody_context
#        self.rhythm_contexts = new_rhythm_context
#
#        if updateMeta:
#            self.update_params(stream.getMetaAnalysis())
#            if self.ins_panel:
#                self.ins_panel.updateMetaParams(self.metaParameters)
#
#    def prepare_meta_data(self):
#        ''' Updates the metaData for the network to use '''
#        values = []
#        for k in sorted(self.metaParameters.keys()):
#            if k == 'ts':
#                frac = Fraction(self.metaParameters[k], _normalize=False)
#                values.extend([frac.numerator, frac.denominator])
#            else:
#                assert isinstance(self.metaParameters[k], (float, int))
#                values.append(self.metaParameters[k])
#
#        self.metaData = np.tile(values, (self.batch_size, 1))



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
        with open('./testData.pkl', 'rb') as f:
            self.music_data = pkl.load(f)

        super().__init__(_id)

        self.current_bar = 0

        self.notes      = self.music_data['instruments'][_id]['melody']['notes']
        self.octaves    = self.music_data['instruments'][_id]['melody']['octaves']
        self.rhythm     = self.music_data['instruments'][_id]['rhythm']

        self.rhythm_contexts = np.array(self.rhythm[0:4])
        self.melody_contexts = np.array([self.notes[0:4]])

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

        # emulate load...
        time.sleep(1)

        #print('Bar generated', bar)
        return bar, None


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

            #print(req)

            _id = req[1]

            if req[0] == 0:
                # new instrument id
                self.return_queues[_id] = req[2]
                #model = DataReader(_id)
                #model = BasicPlayer(_id)
                if len(req) > 3:
                    model = NNPlayer9(_id, req[3])
                else:
                    model = NNPlayer9(_id)

                self.models[_id] = model

                contexts = (model.rhythm_contexts, model.melody_contexts)
                try:
                    rd = model.rhythmDict
                except AttributeError:
                    rd = None

                self.return_queues[_id].put({'init_contexts': contexts, 'rhythm_dict': rd})

            elif req[0] == 1:
                # instument requests new bar
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

            elif req[0] == -1:
                # remove the model and data
                del self.models[_id]
                del self.return_queues[_id]


TEST_MD = {'ts': '4/4', 'span': 10, 'jump': 1.511111111111111, 'cDens': 0.2391304347826087, 'cDepth': 0.0, 'tCent': 62.97826086956522, 'rDens': 1.0681818181818181, 'pos': 0.5, 'expression': 0}
