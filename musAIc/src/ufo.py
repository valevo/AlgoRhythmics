#!/usr/bin/env python3
import sys
import time

import numpy as np
import pickle as pkl
#import music21 as m21

from pythonosc import udp_client, osc_server, dispatcher
from threading import Thread

from keras.models import load_model


class MessageListener(Thread):
    def __init__(self, server):
        super().__init__()
        self.server = server

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()
        #super(MessageListener, self).join()


class Stream():
    '''
    Holds a list of notes and has some useful functions.
    '''
    def __init__(self, notes=None):
        self.stream = dict()
        self.offset_times = []
        self.notes = []

        if notes:
            for n in notes:
                self.append(n)

    def append(self, note):
        '''Adds a Note to the end of the stream'''
        if self.numberOfNotes() > 0:
            self.getLastNote().setNextNote(note)

        offset = note.getOffset()
        self.stream[offset] = note
        self.offset_times.append(offset)

        self.notes.append(note)

    def recalculate(self):
        '''Make sure lists in order and notes all have durations'''
        self.offset_times = sorted(list(self.stream.keys()))
        self.notes = []
        for ot in self.offset_times:
            self.notes.append(self.stream[ot])

    def getStartNote(self):
        '''Returns the start note'''
        return self.stream[self.offset_times[0]]

    def getLastNote(self):
        '''Returns the last note'''
        return self.stream[self.offset_times[-1]]

    def getNextBarNumber(self):
        '''Returns the next bar number'''
        return self.getLastNote().bar + 1

    def show(self):
        '''Prints the stream as a list of notes'''
        for note in self.notes:
            print(note)

    def numberOfNotes(self):
        return len(self.offset_times)

    def toWords(self):
        '''Returns a list of tuples of the form (MIDI_val, duration)'''
        words = []
        self.recalculate()
        for n in self.notes:
            words.append(n.toWord())
        return words

    def appendBar(self, bar):
        '''Append a bar of the form {offset time: MIDI value}'''
        next_bar = self.getNextBarNumber()

        for offset in bar.keys():
            midi = bar[offset]
            if midi == 'EOP':
                continue
            note = Note(midi, next_bar, offset)
            self.append(note)

    def getNotePlaying(self, bar, beat):
        '''Returns the note that is currently playing at bar, beat'''
        offset = self.ts_num * bar + beat
        earlier_notes = [n for n in self.notes if n.offset < offset]
        return earlier_notes[-1]

    def getMetaAnalysis(self):
        '''
        Meta analysis is :
            - span: range between highest and lowest note
            - avgInt: average interval between notes
            - avgChord: proportion of chords
            - tonalCenter: average MIDI value
        '''
        analysis = dict()
        midi_pitches = list([n.midi for n in self.notes])
        span = max(midi_pitches) - min(midi_pitches)
        tonalCenter = sum(midi_pitches) / len(midi_pitches)
        ints = [abs(i-j) for i,j in zip(midi_pitches[:-1], midi_pitches[1:])]
        avgInts = sum(ints) / len(ints)
        avgChord = 0   # for now...

        analysis = {
            'span': span,
            'avgInt': avgInts,
            'avgChord': avgChord,
            'tonalCenter': tonalCenter
        }
        return analysis

    def __len__(self):
        '''Length is the total number of bars'''
        start_bar = self.getStartNote().bar
        end_bar = self.getLastNote().bar
        return end_bar - start_bar + 1


class Note():
    '''
    A Note object holds information on MIDI value, bar number and beat offset,
    plus the next note in the stream.
    '''
    def __init__(self, midi, bar, beat, ts_num=4, ts_den=4, division=2,
                 next_note=None, word=None):

        self.midi = midi
        self.bar = bar
        self.beat = round(division*beat) / division
        self.ts_num = ts_num
        self.ts_den = ts_den
        self.next_note = next_note

    def getOffset(self):
        '''The total offset from the begining of the song'''
        return self.bar * self.ts_num + self.beat

    def getDurationToEnd(self):
        '''Duration to the end of the measure'''
        return self.ts_num - self.beat

    def getDuration(self, next_note=None):
        '''Given the next beat, compute its duration, or until the end of the
        measure'''
        if next_note:
            self.next_note = next_note

        if self.next_note:
            return self.next_note.getOffset() - self.getOffset()
        else:
            return self.getDurationToEnd()

    def setNextNote(self, next_note):
        '''Update the next note'''
        self.next_note = next_note

    def show(self):
        '''Prints the note object'''
        print(self.__str__())

    def toWord(self):
        '''Returns a tuple of the form (MIDI, duration)'''
        return (self.midi, self.getDuration())

    def __str__(self):
        return f'{self.midi} @ {self.bar}:{self.beat} ({ self.getDuration() })'


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
        self.recording = True

        self.beat = 0.0
        self.BPM = 80
        self.BPS = self.BPM / 60
        self.bar = 0

        self.phrase = Stream()
        self.reply = None

        self.player = BasicPlayer()
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

        while not self.input_note:
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



