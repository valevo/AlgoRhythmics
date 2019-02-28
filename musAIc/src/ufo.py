#!/usr/bin/env python3
import sys
import time

import numpy as np
import pickle as pkl
#import music21 as m21

from pythonosc import udp_client, osc_server, dispatcher
from threading import Thread

#from keras.models import load_model

from utils import *


class MessageListener(Thread):
    def __init__(self, server):
        super().__init__()
        self.server = server

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()
        #super(MessageListener, self).join()


class BasicPlayer():
    '''
    Super basic player network, does not coroporate with other musicians or use
    additional information. FOR TESTING ONLY
    '''
    def __init__(self):
        # load model...
        self.model = load_model('./final_model.hdf5')
        self.model._make_predict_function()

        with open('./note_dict.pkl', 'rb') as f:
            self.word_to_id = pkl.load(f)

        self.vocab = len(self.word_to_id)
        self.id_to_word = dict(zip(self.word_to_id.values(), self.word_to_id.keys()))

        self.data = []

        print('Model loaded')

    def initiateModel(self, words):
        data = []
        for w in words:
            data.append(self.wordToID(w))
        while len(data) < 30:
            data += data
        self.data = np.array([data[-30:]])

    def generateBar(self):
        bar = {}
        t = 0
        confidence = 0

        while t < 4:
            prediction = self.model.predict(self.data)
            idxs = np.argsort(-prediction[:, 29, :])[0]
            predict_word = idxs[0 + confidence]
            word = self.id_to_word[predict_word]
            if word == 'EOP':
                # pick next best...
                #predict_word = idxs[1 + confidence]
                #word = self.id_to_word[predict_word]
                bar[t] = word
                break

            bar[t] = word[0]
            t += word[1]
            self.data = np.append(self.data[0][1:], predict_word).reshape((1, 30))

        return bar

    def wordToID(self, word):
        '''Returns the ID of a word for one-hot representation. If not seen
        before, assume ID is 0 (!!!) '''
        try:
            _id = self.word_to_id[word]
        except KeyError:
            _id = 0

        return _id

class DataReader():
    ''' Player that reads in musical data of the Melody, Rhythm, Chords format '''
    def __init__(self):
        with open('./testData.pkl', 'rb') as f:
            self.music_data = pkl.load(f)
        self.current_bar = 0

        self.notes      = self.music_data['instruments'][0]['melody']['notes']
        self.octaves    = self.music_data['instruments'][0]['melody']['octaves']
        self.rhythm     = self.music_data['instruments'][0]['rhythm']

    def generateBar(self):
        ''' Reads the next bar from the file '''

        bar = parseBarData(self.notes[self.current_bar],
                            self.octaves[self.current_bar],
                            self.rhythm[self.current_bar])

        self.current_bar += 1

        print(bar)
        return bar

    def initiateModel(self, data):
        ''' Does not take data as input '''
        pass
        return


class UFO:
    def __init__(self):
        print('Starting...')

        self.disp = dispatcher.Dispatcher()
        self.disp.map('/cc', self.ccRecieve)
        self.disp.map('/noteOn', self.noteOn)

        #self.client = udp_client.SimpleUDPClient('192.168.2.142', 57120)
        self.client = udp_client.SimpleUDPClient('100.75.0.230', 57120)
        self.server = osc_server.BlockingOSCUDPServer(('192.168.56.102', 7121),
                                                      self.disp)
        #self.server = osc_server.BlockingOSCUDPServer(('192.168.56.102', 7121),
                                                      #self.disp)

        self.message_listener = MessageListener(self.server)
        #self.client.send_message('/noteOn', (1, 60))

        self.playing = True
        self.recording = False

        self.beat = 0.0
        self.BPM = 80
        self.BPS = self.BPM / 60
        self.bar = 0

        self.phrase = Stream()
        self.reply = None

        #self.player = BasicPlayer()
        self.player = DataReader()
        self.reply = self.player.generateBar()
        print('AI ready!')

        self.input_note = None
        self.input_cc = None

        self.message_listener.start()
        self.mainLoop()

        # clean up...
        self.message_listener.shutdown()
        self.message_listener.join()


    def ccRecieve(self, *msg):
        if msg[1] == 0:
            return

        if self.recording:
            self.recording = False
            print('\nStopped recording...')
            self.phrase.show()
            print(self.phrase.getMetaAnalysis())
            self.player.initiateModel(self.phrase.toWords())
            self.reply = self.player.generateBar()
            print(self.reply)
        else:
            self.recording = True
            self.reply = None
        #print(self.phrase.toWords())

    def noteOn(self, *msg):
        self.input_note = msg[1]

    def mainLoop(self):
        print('Main loop started!\n')

        print('PLAY NOTE TO START', end='')
        sys.stdout.flush()

        while not self.input_note and self.recording:
            time.sleep(0.01)

        prev_time = time.time()
        met_beat = -1

        reply_bar = None
        prev_note = None

        while self.playing:
            self.beat = (self.beat + (time.time() - prev_time)*self.BPS) % 4.0
            prev_time = time.time()

            # play metronome...
            if met_beat != int(self.beat):
                met_beat = int(self.beat)

                if met_beat == 0:
                    # start of new bar...
                    if self.reply:
                        reply_bar = self.reply
                        self.phrase.appendBar(self.reply)
                        self.reply = self.player.generateBar()

                    self.client.send_message('/noteOn', (1, 40))
                    self.bar += 1
                else:
                    self.client.send_message('/noteOn', (1, 37))

                # print metronome...
                if self.recording:
                    sys.stdout.write(' REC>>>  ' + str(met_beat+1) + '           \r')
                else:
                    sys.stdout.write(' PLAY>>>  ' + str(met_beat+1) + '           \r')

                sys.stdout.flush()

            # check for new notes...
            if self.recording and self.input_note:
                note = Note(self.input_note, self.bar, self.beat)
                self.phrase.append(note)
                self.input_note = None

            elif not self.recording:
                # must be playing, look for new bar...
                if reply_bar:
                    event_times = sorted(list(reply_bar.keys()))
                    if self.beat > event_times[0]:
                        # play the note!
                        if prev_note:
                            self.client.send_message('/noteOff', (2, prev_note))

                        t = event_times[0]
                        note = reply_bar[t]

                        if note == 'EOP':
                            # return to recording...
                            reply_bar = None
                            self.recording = True
                        else:
                            del reply_bar[t]
                            self.client.send_message('/noteOn', (2, note))
                            prev_note = note


            time.sleep(0.005)

        return



if __name__ == '__main__':
    ufo = UFO()



