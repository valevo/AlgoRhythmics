import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from pythonosc import udp_client, osc_server, dispatcher
from threading import Thread


def dist(x, dt):
    dt2 = x%dt
    return min(dt2, dt-dt2)

def loss(dt, events):
    distance = np.array(list(map(lambda x: dist(x, dt), events)))
    return np.sum(distance**2)/dt

def dtToTemp(dt):
    return 60/(dt*8)

def tempoToDt(tempo):
    return 60/(tempo*8)


class MessageListener(Thread):
    def __init__(self, server):
        super().__init__()
        self.server = server

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

class BeatListener:
    def __init__(self):
        self.disp = dispatcher.Dispatcher()
        self.disp.map('/cc', self.ccRecieve)
        self.disp.map('/noteOn', self.noteOn)

        self.client = udp_client.SimpleUDPClient('100.75.0.230', 57120)
        self.server = osc_server.BlockingOSCUDPServer(('192.168.56.102', 7121),
                                                      self.disp)

        self.message_listener = MessageListener(self.server)
        #self.client.send_message('/noteOn', (1, 60))

        self.playing = True
        self.recording = False

        self.phraseNotes = []
        self.phraseTimes = []

        self.input_note = None

        self.message_listener.start()
        self.mainLoop()

        # clean up...
        print('Finishing...')
        self.message_listener.shutdown()
        self.message_listener.join()

    def ccRecieve(self, *msg):
        print('CC: ', msg)
        self.playing = False

    def noteOn(self, *msg):
        self.input_note = msg[1]

    def processInput(self):
        tempos = list(np.arange(60, 141, 1))
        dts = list(map(tempoToDt, tempos))
        losses = list(map(lambda dt: loss(dt, self.phraseTimes), dts))

        print('Found Tempo: ', tempos[np.argmin(losses)])

        plt.plot(tempos, losses)
        plt.plot(tempos, 1/np.array(tempos), c='k', ls='-')
        plt.show()

    def mainLoop(self):
        print('PLAY NOTE TO START')

        while not self.input_note:
            time.sleep(0.01)

        start_time = time.time()
        self.phraseTimes.append(0.0)
        self.phraseNotes.append(self.input_note)
        self.input_note = None

        while self.playing:
            # check for new notes...
            if self.input_note:
                self.phraseTimes.append(time.time() - start_time )
                self.phraseNotes.append(self.input_note)
                self.input_note = None
                sys.stdout.write(str(self.phraseNotes) + '\r')
                sys.stdout.flush()

            time.sleep(0.005)

        self.processInput()

        return

if __name__ == '__main__':
    bl = BeatListener() 























