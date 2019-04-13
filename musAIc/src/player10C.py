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

from v10_dev.Nets.CombinedNetwork import CombinedNetwork
from v10_dev.Nets.MetaPredictor import MetaPredictor
from v10_dev.Nets.ChordNetwork import ChordNetwork

class Player():
    def __init__(self, _id):
        self._id = _id

    def generate_bar(self, **kwargs):
        ''' Requests the next bar '''
        return None

    def get_contexts(self):
        return None

class NNPlayer10C(Player):
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

        trainings_dir = './v10_dev/Trainings/meta_no_embed_wrong_loss/'

        print('Loading Player', _id)

        with open(trainings_dir + 'DataGenerator.conversion_params', 'rb') as f:
            conversion_params = pkl.load(f)
            self.rhythmDict = conversion_params['rhythm']

        self.indexDict = {v: k for k, v in self.rhythmDict.items()}

        metaPredictor = MetaPredictor.from_saved_custom(trainings_dir+'meta/')

        weights_folder = trainings_dir + "weights/_checkpoint_20/"
        self.comb_net = CombinedNetwork.from_saved_custom(weights_folder, metaPredictor,
                                                     generation=True,
                                                     compile_now=False)

        print('Repr:', repr(self.comb_net))

        self.context_size = self.comb_net.params["context_size"]

        #print(self.comb_net.params)
        self.V_rhythm = self.comb_net.params["rhythm_net_params"][2]
        self.m = self.comb_net.params["melody_bar_len"]
        self.V_melody = self.comb_net.params["melody_net_params"][3]

        self.batch_size = 1
        self.bar_length = 4

        # load chord network...
        with open(trainings_dir + 'ChordGenerator.conversion_params', 'rb') as f:
            chord_conv_params = pkl.load(f)
            self.chordDict = chord_conv_params['chords']
            self.indexChordDict = {v: k for k, v in self.chordDict.items()}

        self.chord_net = ChordNetwork.from_saved_custom(trainings_dir+'/chord/', load_melody_encoder=True)


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
            notes = np.array([1, 3, 5, 6, 8, 10, 12])
            self.melody_contexts = rand.choice(list(notes)+list(notes+12),
                                               size=(self.batch_size, self.context_size, self.m))
        elif scale == 'min':
            notes = np.array([1, 3, 4, 6, 8, 10, 11])
            self.melody_contexts = rand.choice(list(notes)+list(notes+12),
                                               size=(self.batch_size, self.context_size, self.m))
        elif scale == 'pen':
            notes = np.array([1, 4, 6, 8, 11])
            self.melody_contexts = rand.choice(list(notes)+list(notes+12),
                                               size=(self.batch_size, self.context_size, self.m))
        elif scale == '5th':
            notes = np.array([1, 8])
            self.melody_contexts = rand.choice(list(notes)+list(notes+12),
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

        print('Sampled rhythm:', [self.indexDict[r] for r in sampled_rhythm[0]])

        if 'loopRhythm' in kwargs:
            if kwargs['loopRhythm'] > 0:
                # ignore the sampled rhythm
                print('loopRhythm')
                num = min(len(self.rhythm_contexts), kwargs['loopRhythm'])
                if num > 0:
                    sampled_rhythm = self.rhythm_contexts[-num]

        # -- UPDATE CONTEXTS
        if 'um' in kwargs:
            um = kwargs['um']
        else:
            um = 2
        update_mode = {0: 'none',
                       1: 'top',
                       2: 'sampled',
                       3: 'inject',
                       4: 'user'}[um]


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
        else:
            pass

        print('lead_mode:', lead_mode, ' sample_mode:', sample_mode, ' update_mode:', update_mode)

        #print(sampled_rhythm)
        #print(sampled_melody)

        # convert to bar...
        rhythm = [self.indexDict[b] for b in sampled_rhythm[0]]
        if chord_num == 0:
            # use chord generator network
            melody = []
            for n in sampled_melody[0]:
                print(n, end=', ')
                if n >= 12:
                    chord_outputs = self.chord_net.predict(
                        x=[np.array([[n]]), self.melody_contexts[:, -1:, :], self.embeddedMetaData]
                    )
                    if sample_mode == 'argmax':
                        chord = np.argmax(chord_outputs[0], axis=-1)
                    elif sample_mode == 'dist' or sample_mode == 'top':
                        chord = rand.choice(len(chord_outputs[0]), p=chord_outputs[0])

                    print('CHORD:\n', chord_outputs, '\n', chord_outputs.shape)
                    print('Selected chord:', chord, self.indexChordDict[chord])

                    intervals = self.indexChordDict[chord]
                    #intervals = [0, 4, 7]
                    melody.append([n + i - 12 for i in intervals])

                else:
                    melody.append(n)

            print()

        elif chord_num == 1:
            # no chords, just melody
            melody = [int(n) for n in sampled_melody[0]]
        else:
            # use sample chords
            melody = sampled_chords

        # TODO: smarter octave choices
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

        self.embeddedMetaData = metaData

TEST_MD = {'ts': '4/4', 'span': 10, 'jump': 1.511111111111111, 'cDens': 0.2391304347826087, 'cDepth': 0.0, 'tCent': 62.97826086956522, 'rDens': 1.0681818181818181, 'pos': 0.5, 'expression': 0}
