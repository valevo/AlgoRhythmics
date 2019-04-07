import tkinter as tk
from tkinter import font
from tkinter.ttk import Style
import threading
import multiprocessing
import queue
import time
from copy import deepcopy
from collections import deque, defaultdict

from guielements import COLOR_SCHEME, VScrollFrame, InstrumentPanel, PlayerControls, Knob
import utils
from simpleDialog import Dialog
from networkEngine import NetworkManager
from pythonosc import udp_client, osc_server, dispatcher

import numpy as np

# Address and port of host machine to send messages to OSC-MIDI bridge:
# ON WINDOWS:
#   - run ipconfig in command prompt for Address (Wireless Adapter WiFi)
#   - run NetAddr.localAddr; in SuperCollider to get port number
#CLIENT_ADDR = '192.168.0.31'
#CLIENT_ADDR = '146.50.249.176'
#CLIENT_ADDR = '192.168.0.36'
#CLIENT_ADDR = '145.109.5.179'
CLIENT_ADDR = '100.75.0.230'
#CLIENT_ADDR = '127.0.0.1'
CLIENT_PORT = 57120


# Address and port of guest machine to listen for incoming messages:
#   - run ifconfig (or `ip addr` in Arch) and use Host-Only IP (same as listed on Windows ipconfig)
#   - port can be arbitrary, but SuperCollider must know!
#SERVER_ADDR = '192.168.56.102'
SERVER_ADDR = '192.168.56.102'
#SERVER_ADDR = '127.0.0.1'
SERVER_PORT = 7121

TOUCH_OSC_CLIENT = ('192.168.0.32', 8000)  # Address that TouchOSC is recieving messages from (check app)
TOUCH_OSC_SERVER = ('10.0.2.15', 7121)       # Address from where TouchOSC messages will arrive from 
                                             #   (check NAT, Port Forwarding in VBox)

META_PARAMS = ['span', 'tCent', 'cDens', 'cDepth', 'jump', 'rDens', 'sPos']

APP_NAME = 'musAIc LIVE (v0.6)'


class MessageListener(threading.Thread):
    """Listens to incoming OSC messages"""
    def __init__(self, server):
        threading.Thread.__init__(self)
        self.server = server

    def run(self):
        self.server.serve_forever()

    def join(self, timeout=None):
        self.server.shutdown()
        super(MessageListener, self).join(timeout)

class Clock(multiprocessing.Process):
    ''' The Clock that keeps time, updates 24 times per beat '''
    def __init__(self, client, clockVar):
        super(Clock, self).__init__()

        self.client = client
        self.clockVar = clockVar
        self.clockVar['bar'] = 0
        self.clockVar['beat'] = 0.0
        self.clockVar['tick'] = 0
        self.clockVar['playing'] = False
        self.clockVar['stopping'] = False
        self.clockVar['recording'] = False

        self.bar = 0
        self.beat = 0.0

        self.playEvent = multiprocessing.Event()
        self.stop_request = multiprocessing.Event()

        self.stop = False

    def run(self):
        while not self.stop_request.is_set():
            if self.playEvent.is_set():
                bpm = self.clockVar['bpm']
                tick_time = (60/bpm)/24

                clock_on = time.time()
                for i in range(24 * 4):
                    next_time = clock_on + i*tick_time
                    while time.time() < next_time:
                        pass
                    # send MIDI clock...
                    self.client.send_message('/clock', 1)

                    # update clockVar...
                    self.beat = i/24
                    self.clockVar['beat'] = self.beat
                    self.clockVar['tick'] = i

                    if self.stop_request.is_set():
                        return

                if self.stop:
                    self.bar = 0
                    self.stop = False
                else:
                    self.bar += 1
                self.clockVar['bar'] = self.bar
                self.clockVar['beat'] = 0.0
                #self.clockVar['tick'] = 0

            else:
                # not playing
                time.sleep(0.01)


    def join(self, timeout=None):
        self.stop_request.set()
        #self.wait_event.set()
        self.playEvent.clear()
        super(Clock, self).join(timeout)

    def set_playing(self):
        if self.playEvent.is_set():
            return
        self.clockVar['playing'] = True
        self.client.send_message('/clockStart', 1)
        self.playEvent.set()

    def toggle_playback(self):
        if self.playEvent.is_set():
            self.playEvent.clear()
            self.clockVar['playing'] = False
            self.clockVar['stopping'] = False
        else:
            self.playEvent.set()
            self.clockVar['playing'] = True
            self.clockVar['stopping'] = False
            self.client.send_message('/clockStart', 1)
            print('clock start')

    def toggle_stop(self, stop_button=None):
        if self.playEvent.is_set():
            if not self.stop:
                self.stop = True
                self.playEvent.clear()
                self.clockVar['playing'] = False
                self.clockVar['stopping'] = True
            else:
                self.stop = False
                self.playEvent.set()
                self.clockVar['playing'] = True
                self.clockVar['stopping'] = False
        else:
            self.stop = False
            self.bar = 0
            self.clockVar['bar'] = self.bar
            self.clockVar['beat'] = 0.0
            self.clockVar['playing'] = False
            self.clockVar['stopping'] = False


class Engine2(threading.Thread):
    ''' Updates instruments and send OSC messages '''
    def __init__(self, app, client, clockVar, ins_manager):
        super(Engine2, self).__init__()

        self.app = app
        self.client = client
        self.ins_manager = ins_manager
        self.clockVar = clockVar

        self.stop_request = multiprocessing.Event()

        self.current_bar = 0
        self.stop_flag = False

        #self.next_events = defaultdict(defaultdict)       # {id: {time % 4.0: [msgs]}}

    def run(self):
        while not self.stop_request.is_set():
            while self.clockVar['playing']:
                # clock is playing...

                # -- START OF MEASURE
                self.stop_flag = True
                events = []
                for ins in self.ins_manager.get_instruments():
                    ins.load_bar()

                    new_events = ins.get_bar_events()
                    events.append(new_events)

                last_tick = -1

                while self.clockVar['bar'] == self.current_bar:

                    # -- DURING THE MEASURE
                    if self.clockVar['tick'] != last_tick:
                        last_tick = self.clockVar['tick']
                        for event in events:
                            for addr, args in event[last_tick]:
                                print(addr, args)
                                self.client.send_message(addr, args)
                    else:
                        time.sleep(0.001)

                # -- END OF MEASURE
                self.current_bar = self.clockVar['bar']
                for ins in self.ins_manager.get_instruments():
                    ins.close_bar()
                    ins.increment_bar_num()

                self.ins_manager.check_generate_bars()
                self.ins_manager.update_ins()


            else:
                # clock is stopped...
                if self.stop_flag:
                    self.stop_flag = False
                    self.client.send_message('/clockStop', 1)

                    if self.clockVar['stopping']:
                        self.clockVar['stopping'] = False
                        for ins in self.ins_manager.get_instruments():
                            ins.reset()
                        self.next_events = defaultdict(defaultdict)


                self.ins_manager.check_generate_bars()
                self.ins_manager.update_ins()
                time.sleep(0.001)

    def join(self, timeout=None):
        self.stop_request.set()
        super(Engine2, self).join(timeout=None)


class Engine(multiprocessing.Process):
    """Controls the clock and instument playback"""
    def __init__(self, app, ins_manager, bpmVar, touchOSCClient=None):
        #threading.Thread.__init__(self)
        super(Engine, self).__init__()

        self.app = app
        self.ins_manager = ins_manager
        #self.bpmVar = bpmVar
        self.touchOSCClient = touchOSCClient

        self.stop_request = multiprocessing.Event()
        #self.play_request = multiprocessing.Event()
        self.beat = 0
        self.bar = 0
        self.start_playing = True
        self.record = False
        self.stop = False

    def run(self):
        while not self.stop_request.is_set():
            if self.play_request.is_set():
                # start bar process...
                if self.start_playing:
                    time_bar_start = time.time()
                    self.start_playing = False
                    #if self.record:
                    #    self.app.sendCC(1, 127)

                # before bar processes...
                for ins in self.ins_manager.get_instruments():
                    ins.load_bar()

                try:
                    new_bps = self.bpmVar.get()/60
                    bps = new_bps
                except:
                    pass

                # during bar processes...
                clock_on = time.time()
                while self.beat < 3.98 and not self.stop_request.is_set():
                    self.ins_manager.send_message('/clock', 1)
                    self.beat = min(3.99, bps*(time.time() - time_bar_start))
                    #self.ins_manager.play_instruments(self.bar, self.beat)

                    if self.touchOSCClient:
                        self.touchOSCClient.send_message('/metronome', tuple([1 if n < self.beat else 0 for n in range(4)]))

                    wait_time = clock_on - time.time() + 1/(bps*24)
                    print(wait_time)
                    clock_on = time.time()
                    #self.wait_event.wait(timeout=wait_time)
                    time.sleep(max(0, wait_time))

                # after bar processes...
                for ins in self.ins_manager.get_instruments():
                    ins.close_bar()
                    ins.increment_bar_num()

                self.bar += 1
                self.beat = 0.0
                time_bar_start += 4/bps
            else:
                # after stopping playback...
                #self.is_playing.clear()
                if not self.start_playing:
                    self.ins_manager.send_message('/clockStop', 1)
                    if self.touchOSCClient:
                        self.touchOSCClient.send_message('/metronome', (0, 0, 0, 0))
                    #self.app.sendCC(1, 0)
                    #self.app.sendCC(2, 127)
                    self.record = False

                    if self.stop:
                        print('resetting positions')
                        self.stop = False
                        for ins in self.ins_manager.get_instruments():
                            ins.reset()
                        self.bar = 0
                        self.ins_manager.load_nextbar()

                    self.app.controls.update_buttons()
                    self.start_playing = True
                    self.ins_manager.set_stopped()

            # update instruments if they have requested new bars...
            #for ins in self.ins_manager.get_instruments():
            #    ins.check_updates()

            #self.play_request.wait(timeout=0.05)

    def join(self, timeout=None):
        self.stop_request.set()
        #self.wait_event.set()
        self.play_request.clear()
        super(Engine, self).join(timeout)

    def toggle_playback(self):
        print('toggle playback')
        if self.play_request.is_set():
            self.play_request.clear()
        else:
            self.play_request.set()
            self.ins_manager.send_message('/clockStart', 1)
            print('clock start')
        print(self.play_request.is_set())

    def toggle_stop(self, stop_button=None):
        if not self.stop:
            self.stop = True
            self.play_request.clear()
        else:
            self.stop = False
            self.play_request.set()


PAUSED = 2          # currently paused and not playing
PLAY_WAIT = -1      # currently paused but will play at next bar
PAUSE_WAIT = -2     # currently playing but will pause at end of bar
PLAYING = 1         # currently playing


class Instrument():
    def __init__(self, chan, ins_id, ins_manager, request_queue, network_queue):
        self.chan = chan
        self.ins_id = ins_id
        self.ins_manager = ins_manager
        self.ins_panel = None
        self.request_queue = request_queue   # queue to send new bar requests
        self.network_queue = network_queue   # queue to recieve new bars

        self.transpose = 0                # number of octaves to shift output
        self.confidence = 0               # the rank of which note to play
        self.bar_num = 0                  # current bar number
        self.global_bar_num = 0           # global bar number
        self.stream = utils.Stream()      # the musical stream of notes / chords
        self.context_stream = dict()      # the bar in 'context' form
        self.last_requested_bar = -1      # requested bars
        self.context_size = 4             # number of contexts needed
        self.status = PAUSED              # current playing status
        self.armed = False                # is armed (recording) instrument
        self.active = True                # is selected instrument
        self.mute = False                 # mute the output
        self.continuous = True            # continuously generate new bars
        self.recording = False            # if instrument is recording input
        self.record_bars = []             # range of recording bars [start, end]

        self.lead = False                 # Lead instrument that others follow

        self.next_note = None             # the current note object
        self.offMode = 0                  # note off mode: 0 - hold to noteOn
                                          #               >0 - hold for this time
        self.heldNotes = {}               # a dict of current notes being held, {MIDI: note}
        self.prev_events = defaultdict(list)

        self.hold = False                 # holds current context

        self.request_GUI_update = True    # lets GUI thread know it needs to recalc canvas

        self.loopLevel = 0                # number of bars to loop 
        self.loopEnd = 0                  # bar the loop number offsets from
        self.rhythmLoopLevel = 0          # what level to loop the rhythm information in generation

        # SHOULD BE QUEUE?
        self.noteOn = None                # any noteOn events placed here
        self.recorded_bar = {}            # currently recorded bar

        if self.ins_id == 0:
            self.status = PLAYING

        self.check_updates()

        # request some bars
        for _ in range(4):
            self.request_new_bar()

    def set_ins_panel(self, ins_panel):
        self.ins_panel = ins_panel
        self.update_params(self.ins_panel.getMetaParams())

    def delete(self):
        '''Deletes this instrument'''
        self.ins_manager.removeInstrument(self.ins_id)

    def increment_bar_num(self):
        if self.status == PAUSED:
            return
        else:
            self.bar_num += 1

    def load_bar(self):
        '''Loads the current bar into memory to be played, called at start of bar'''
        if self.status == PAUSED or self.status == PAUSE_WAIT:
            return

        if self.status == PLAY_WAIT:
            self.status = PLAYING

        if self.loopLevel > 0:
            if self.bar_num > self.loopEnd:
                self.bar_num = self.loopEnd - self.loopLevel + 1
                self.bar_num = max(0, self.bar_num)
                self.next_note = self.stream.getNextNotePlaying(self.bar_num, 0)

        print('LOAD BAR', self.bar_num)

        if self.record_bars:
            if self.bar_num >= self.record_bars[0] and \
               self.bar_num < self.record_bars[1]:
                self.recording = True
        else:
            self.recording = False
            try:
                self.current_bar = self.stream.getBar(self.bar_num)
                if not self.next_note:
                    self.next_note = self.stream.getNextNotePlaying(self.bar_num, 0)

                #print(self.next_note)

            except Exception as e:
                # no bar left to load...
                print(e)
                self.current_bar = None
                self.next_note = None


    def play_bar(self, bar, beat):
        '''Checks if have to play any notes in current bar'''
        #print('PLAY BAR', self.bar_num, bar, beat)
        #self.check_updates()

        self.global_bar_num = bar

        if self.status == PAUSED or self.mute or self.status == PLAY_WAIT:# or self.current_bar == None:
            return

        if self.recording:
            #print('REC: ', bar, beat)
            # parse any inputs recieved, something like...
            if self.noteOn:
                print('play_bar: noteOn', self.noteOn)
                self.recorded_bar[round(beat*8)/8] = self.noteOn
                self.noteOn = None
            return

        # check if have to mute any notes...
        muted = []
        curr_offset = self.bar_num*4 + beat
        for m, n in self.heldNotes.items():
            if self.offMode < 0.1:
                # hold to noteOn
                offTime = n.getOffset() + n.getDuration()
                if curr_offset > offTime:
                    self.mute_note(n)
                    muted.append(m)

            else:
                # hold for specific time
                offTime = n.getOffset() + self.offMode
                if curr_offset > offTime:
                    self.mute_note(n)
                    muted.append(m)

        for m in muted:
            del self.heldNotes[m]

        if self.next_note:
            if beat < self.next_note.beat or self.bar_num != self.next_note.bar:
                return

            if self.next_note.midi < 0:
                # rest...
                self.next_note = self.next_note.next_note
                return

            #self.mute_last_notes(beat)
            note = self.next_note.midi + 12*self.transpose
            vel = np.random.randint(70, 100)
            if self.next_note.isChord():
                for c in self.next_note.chord:
                    cNote = note + c
                    self.ins_manager.send_message('/noteOn', (self.chan, cNote, vel))
                    #print('Play note {} on channel {}'.format(cNote, self.chan))

            else:
                self.ins_manager.send_message('/noteOn', (self.chan, note, vel))
                print('Play note {} on channel {}'.format(note, self.chan))

            self.heldNotes[self.next_note.midi] = self.next_note
            self.next_note = self.next_note.next_note

    def get_bar_events(self):
        ''' returns a dict of {tick: MIDI event} for the current bar'''

        if self.status == PAUSED or self.mute or self.status == PLAY_WAIT or self.current_bar == None:
            return None

        events = deepcopy(self.prev_events)
        self.prev_events = defaultdict(list)

        for offset, note in self.current_bar.items():
            tick = int(offset*24)
            vel = np.random.randint(70, 100)
            events[tick].append(('/noteOn', (self.chan, note.midi, vel)))
            if self.offMode > 0.1:
                offTick = tick + int(self.offMode*24)
            else:
                offTick = tick + int(note.getDuration()*24)

            if offTick >= 24 * 4:
                self.prev_events[offTick % 4.0].append(('/noteOff', (self.chan, note.midi)))
            else:
                events[offTick].append(('/noteOff', (self.chan, note.midi)))

        return events


    def mute_note(self, note):
        #print('Muting note ', note)
        m = note.midi
        self.ins_manager.send_message('/noteOff', (self.chan, m))
        if note.isChord():
            for c in note.chord:
                cNote = m + c
                self.ins_manager.send_message('/noteOff', (self.chan, cNote))

    def mute_all_notes(self):
        self.ins_manager.send_message('/allOff', self.chan)

    def request_new_bar(self, lead_bar=None):
        #diff = self.stream.getNextBarNumber() - self.bar_num - 1
        #print('Requesting diff {}'.format(diff))

        #lead_contexts = None
        #if not self.lead:
        #    try:
        #        lead_contexts = self.ins_manager.lead_ins.get_contexts(diff)
        #    except AttributeError:
        #        print('No lead instrument')
        #        pass

        msg = {
            'confidence':       self.confidence,
            'loopRhythm':       self.rhythmLoopLevel,
            'hold':             self.hold,
            'lead_bar':         lead_bar
        }
        self.request_queue.put((1, self.ins_id, msg))
        self.last_requested_bar += 1

    def check_updates(self):
        '''Checks if any messages from network'''
        try:
            while not self.network_queue.empty():
                msg = self.network_queue.get(block=False)

                if 'init_contexts' in msg:
                    rhythm_contexts, melody_contexts = msg['init_contexts']
                    self.context_size = len(rhythm_contexts)

                    for i in range(self.context_size):
                        c = (rhythm_contexts[i], melody_contexts[0, i, :])
                        self.context_stream[i-self.context_size] = c

                    print('init_contexts updated')

                if 'bar' in msg:
                    if self.recording or self.armed:
                        return
                    new_bar = msg['bar']
                    context = msg['context']
                    self.stream.appendBar(new_bar)
                    b_num = self.stream.getLastNote().bar

                    self.context_stream[b_num] = context

                    self.request_GUI_update = True

                if 'md' in msg:
                    self.ins_panel.updateMetaParams(msg['md'])

        except Exception as e:
            # no new bars...
            print('Caught Exception in check_updates() for ins_id {}:'.format(self.ins_id))
            print(e)
            return

    def close_bar(self):
        '''Called at end of bar'''
        print('CLOSE BAR', self.bar_num)
        if self.status == PAUSE_WAIT:
            self.status = PAUSED

        if self.bar_num == len(self.stream) - 1 and not self.continuous:
            self.status = PAUSED

        if self.recording:
            print('close bar, self.recording=True')
            # only update bar if anything was recorded
            print('Recorded bar:', self.recorded_bar)

            if len(self.recorded_bar) == 0:
                self.recorded_bar = {0.0: -1}
            self.stream.appendBar(self.recorded_bar)
            self.recorded_bar = {}

            if self.bar_num == self.record_bars[-1] - 1:
                print('FINISHED RECORDING')
                self.recording = False
                self.init_record(self.bar_num, 0)
                self.stream.recalculate()
                self.stream.show()
                self.request_GUI_update = True
                #self.ins_panel.update_canvas()
                self.ins_panel.update_highlighted_bars()
                self.request_queue.put((2, self.ins_id, self.stream))

                # if live update of meta parameters...
                self.ins_panel.updateMetaParams(self.stream.getMetaAnalysis())


    def reset(self):
        ''' Called when player is stopped '''
        self.mute_all_notes()
        self.bar_num = 0
        for n in self.stream.notes:
            n.played = False
        self.next_note = self.stream.getNextNotePlaying(self.bar_num, 0)

    def init_record(self, start, num):
        if num == 0:
            self.record_bars = []
            self.armed = False
            self.ins_panel.recVar.set(0)
            self.ins_panel.update_highlighted_bars()
        else:
            print('init_record')
            self.record_bars = [start, start+num]
            self.armed = True
            self.recorded_bar = {}

            # clear the rest of the stream...
            self.stream.removeBarsToEnd(start)
            self.ins_panel.clear_canvas(start)
            self.ins_panel.update_highlighted_bars()

        print(self.record_bars)

    def get_contexts(self, diff):
        ''' Returns the contexts needed to generate the future DIFF bar '''
        req_bar = self.bar_num + diff
        #print('requested bar:', req_bar)
        #print(self.context_stream)

        if len(self.context_stream) < diff:
            print('Not enough context information... (diff = {})'.format(diff))
            return None

        if req_bar not in self.context_stream:
            print('Requested a bar too far in the future (diff = {})'.format(diff))
            return None

        else:
            result = []
            for i in range(req_bar - self.context_size, req_bar+1):
                try:
                    #print('Context updated for bar {}'.format(i))
                    c = self.context_stream[i]
                except KeyError:
                    print('Unknown bar index {}, using latest'.format(i))
                    key = max(self.context_stream.keys())
                    c = self.context_stream[key]

                result.append(c)

        return result

    def update_params(self, parameters):
        self.request_queue.put((3, self.ins_id, parameters))
        for p, v in parameters.items():
            self.ins_manager.send_touchOSC_message('/{}/{}'.format(self.ins_id+1, p), (v))
            #print('/{}/{}'.format(self.ins_id, p), (v))

    def toggle_paused(self):
        if self.status == PAUSED or self.status == PAUSE_WAIT:
            self.status = PLAY_WAIT
            self.ins_manager.send_touchOSC_message('/{}/muteLED'.format(self.ins_id+1), (0))
        else:
            self.ins_manager.send_touchOSC_message('/{}/muteLED'.format(self.ins_id+1), (1))
            self.status = PAUSE_WAIT

    def toggle_loop(self, level):
        if self.loopLevel == level:
            self.loopLevel = 0
            #self.ins_manager.send_touchOSC_message('/{}/loopLED/{}'.format(self.ins_id, level), 0)
        else:
            self.loopLevel = level
            self.loopEnd = self.bar_num
            #self.ins_manager.send_touchOSC_message('/{}/loopLED/{}'.format(self.ins_id, level), 1)

        #print(self.loopLevel, self.loopEnd)
        self.ins_panel.update_highlighted_bars()

    def toggle_rhythm_loop(self, level):
        if self.rhythmLoopLevel == level:
            self.rhythmLoopLevel = 0
        else:
            self.rhythmLoopLevel = level
        print('Rhythm loop level set to ', self.rhythmLoopLevel)

    def toggle_mute(self):
        self.mute = not self.mute

    def toggle_continuous(self):
        self.continuous = not self.continuous

    def toggle_hold(self):
        self.hold = not self.hold
        print('HOLD set to ', self.hold)
        self.ins_manager.send_touchOSC_message('/{}/holdLED'.format(self.ins_id+1), int(self.hold))


class InstrumentManager():
    """
    Holds all the instruments and manages their processes
    Midi player and GUI threads use this
    """
    def __init__(self, ins_box, client, request_queue, queue_manager, touchOSCClient=None):
        self.instruments = {}               # contains the instruments by ID
        self.instrumentPanels = {}          # the GUI display panels
        self.ins_box = ins_box              # where the panels are displayed
        self.client = client                # the client that sends the played notes
        self.request_queue = request_queue  # requests to network Engine
        self.queue_manager = queue_manager  # holds the instrument specific queues
        self.ins_counter = 0                # last instrument ID
        self.armed_ins = None               # if any instruments set to record
        self.selected_ins = None            # the currently selected instrument
        self.lead_ins = None                # the lead instrumet
        self.touchOSCClient = touchOSCClient

    def addInstrument(self):
        print('adding instrument {}...'.format(self.ins_counter))
        # create the new return queue 
        return_queue = self.queue_manager.Queue()
        self.request_queue.put((0, self.ins_counter, return_queue))
        instrument = Instrument(chan=self.ins_counter+1,
                                ins_id=self.ins_counter, ins_manager=self,
                                network_queue=return_queue,
                                request_queue=self.request_queue)

        insPanel = InstrumentPanel(self.ins_box, instrument)
        instrument.set_ins_panel(insPanel)

        self.instrumentPanels[self.ins_counter] = insPanel
        self.instruments[self.ins_counter] = instrument
        self.ins_counter += 1
        insPanel.pack(side='bottom', fill='x', pady=5)
        #self.ins_box.onFrameConfigure()

        self.set_selected_instrument(instrument)
        if not self.lead_ins:
            self.set_lead_instrument(instrument)

    def removeInstrument(self, ins_id):
        if self.selected_ins.ins_id == ins_id:
            try:
                next_id = list(self.instruments.keys())[0]
                self.set_selected_instrument(self.instruments[next_id])
            except:
                self.selected_ins = None

        if self.lead_ins.ins_id == ins_id:
            try:
                next_id = list(self.instruments.keys())[0]
                self.set_lead_instrument(self.instruments[next_id])
            except:
                self.lead_ins = None

        del self.instruments[ins_id]
        self.instrumentPanels[ins_id].destroy()
        del self.instrumentPanels[ins_id]
        self.request_queue.put((-1, ins_id))
        print('Removing instrument {}'.format(ins_id))
        #self.ins_box.onFrameConfigure()

    def update_ins(self):
        #print('update_ins')
        for ins in self.instruments.values():
            ins.check_updates()

    def check_generate_bars(self):
        # first generate lead instruments bar, then pass to others

        if self.lead_ins:
            # check if lead bar need to generate...
            if self.lead_ins.last_requested_bar - self.lead_ins.bar_num < 6:
                self.lead_ins.request_new_bar()

        else:
            return

        # only generate other bars if lead has the bar already...
        #lead_bar_num_diff = self.lead_ins.bar_num - self.lead_ins.
        lead_last_bar_num = len(self.lead_ins.stream)

        for ins in self.instruments.values():
            if ins.lead:
                continue

            bar_diff = ins.stream.getNextBarNumber() - ins.bar_num
            if bar_diff < 6:
                requesting_bar_num = ins.stream.getNextBarNumber()
                if requesting_bar_num <= ins.last_requested_bar:
                    continue

                if lead_last_bar_num >= self.lead_ins.bar_num + bar_diff:
                    context = self.lead_ins.context_stream[self.lead_ins.bar_num + bar_diff]
                    ins.request_new_bar(lead_bar=context)
                else:
                    continue


    def load_nextbar(self):
        for _id in self.instruments.keys():
            self.instruments[_id].load_bar()

    def play_instruments(self, bar, beat):
        for ins in self.get_instruments():
            #if ins.status == PLAYING or ins.status == PAUSE_WAIT:
            ins.play_bar(bar, beat)

    def get_instruments(self):
        return list(self.instruments.values())

    def get_leads_contexts(self):
        ''' Returns the lead's rhythm and melody contexts '''
        return self.lead_ins.get_contexts()

    def set_recording_instrument(self, ins, num):
        if self.armed_ins:
            self.armed_ins.init_record(0, 0)
            self.instrumentPanels[self.armed_ins.ins_id].recVar.set(0)

        if num == 0:
            self.armed_ins = None
        else:
            self.armed_ins = ins
            if self.armed_ins.status == PLAYING:
                self.armed_ins.init_record(self.armed_ins.bar_num + 2, num)
            else:
                self.armed_ins.init_record(self.armed_ins.bar_num, num)

            self.instrumentPanels[self.armed_ins.ins_id].recVar.set(num)

    def set_playing(self):
        for ins in self.instruments.values():
            ins.status = PLAYING

    def set_stopped(self):
        pass
        #for ins in self.instruments.values():
        #    ins.status = PLAY_WAIT
        #    pass

    def set_lead_instrument(self, ins):
        print('Setting lead ins to', ins)
        if self.lead_ins:
            self.lead_ins.lead = False
            self.lead_ins.ins_panel.updateLead()

        self.lead_ins = ins
        if ins:
            self.lead_ins.lead = True
            self.lead_ins.ins_panel.updateLead()

    def set_selected_instrument(self, ins):
        if self.selected_ins:
            self.selected_ins.active = False
            self.selected_ins.ins_panel.configure(highlightbackground='gray', highlightthickness=1)

        self.selected_ins = ins
        if ins:
            self.selected_ins.active = True
            self.selected_ins.ins_panel.configure(highlightbackground='orange', highlightthickness=1)

    def select_up(self):
        all_ids = list(self.instruments.keys())
        if len(all_ids) > 0:
            if self.selected_ins:
                selected_id = self.selected_ins.ins_id
                try:
                    new_id = min([a for a in all_ids if a > selected_id])
                except:
                    return
            else:
                new_id = min(all_ids)
        else:
            return

        self.set_selected_instrument(self.instruments[new_id])

    def select_down(self):
        all_ids = list(self.instruments.keys())
        if len(all_ids) > 0:
            if self.selected_ins:
                selected_id = self.selected_ins.ins_id
                try:
                    new_id = max([a for a in all_ids if a < selected_id])
                except:
                    return
            else:
                new_id = max(all_ids)
        else:
            return

        self.set_selected_instrument(self.instruments[new_id])

    def send_message(self, addr, msg):
        try:
            self.client.send_message(addr, msg)
        except OSError:
            print('[OS Error] Instrument manager cannot send message {}:{}'.format(
                      addr, msg))

    def send_touchOSC_message(self, addr, msg):
        try:
            if self.touchOSCClient:
                self.touchOSCClient.send_message(addr, msg)
        except OSError:
            print('[OS Error] Instrument manager cannot send TouchOSC message {}:{}'.format(
                      addr, msg))


class MusaicApp():
    client = None
    server = None

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_NAME)
        self.root.config(bg=COLOR_SCHEME['panel_bg'])
        hs = self.root.winfo_screenheight()
        self.root.geometry('1200x600+0+%d'%(hs/2))
        #self.root.resizable(0, 0)

        self.style = Style()
        self.style.theme_use('clam')

        self.cl_ip = CLIENT_ADDR
        self.cl_port = CLIENT_PORT
        self.sv_ip = SERVER_ADDR
        self.sv_port = SERVER_PORT

        #self.client = udp_client.SimpleUDPClient(self.cl_ip, self.cl_port)
        self.updateClient()

        # for TouchOSC app...
        self.touchOSCdisp = dispatcher.Dispatcher()
        self.touchOSCdisp.map('/*', self.parseOSC)

        self.touchOSCServer = osc_server.BlockingOSCUDPServer(TOUCH_OSC_SERVER, self.touchOSCdisp)
        self.touchOSCClient = udp_client.SimpleUDPClient(*TOUCH_OSC_CLIENT)

        # setup managers...
        self.queue_manager = multiprocessing.Manager()
        self.request_queue = self.queue_manager.Queue()
        self.network_manager = NetworkManager(self.request_queue)

        self.network_manager.start()

        self.mainframe = tk.Frame(self.root)
        self.mainframe['bg'] = COLOR_SCHEME['dark_grey']
        self.mainframe.pack_propagate(False)
        self.mainframe.pack(fill='both', expand=True)
        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.rowconfigure(0, weight=1)

        # main control panel...
        self.maincontrols = tk.Frame(self.mainframe, bg=self.mainframe.cget('bg'))

        timeFont = font.Font(family='Courier', size=12, weight='bold')
        self.timeLabel = tk.Label(self.maincontrols, text='00:00', bg='#303030', fg='yellow',
                             width=10, relief='sunken', bd=2,
                             font='Arial 16', padx=5, pady=5)

        self.BPM = tk.IntVar(self.maincontrols)
        self.bpmBox = tk.Spinbox(self.maincontrols, from_=20, to=240, textvariable=self.BPM,
                            width=3, bg=self.maincontrols.cget('bg'), fg=COLOR_SCHEME['text_light'],
                            buttonbackground=self.maincontrols.cget('bg'))

        self.panicButton = tk.Button(self.maincontrols, text='Panic',
                                     command=self.panic, bg=self.maincontrols.cget('bg'),
                                     fg=COLOR_SCHEME['text_light'])
        # pack main controls...
        self.timeLabel.grid(row=0, column=4, rowspan=2, padx=5)
        tk.Label(self.maincontrols, text='Tempo:', fg=COLOR_SCHEME['text_light'],
                 bg=self.maincontrols.cget('bg')).grid(row=0, column=5, sticky='e')
        self.bpmBox.grid(row=0, column=6)
        self.panicButton.grid(row=1, column=5, columnspan=2, sticky='ew')

        # add instrument button and panels...

        self.instrumentsBox = tk.Frame(self.mainframe, relief='sunken', bd=2, bg=COLOR_SCHEME['note_panel_bg'])
        #self.instrumentsBox = VScrollFrame(self.mainframe, relief='sunken', bd=2,)

        self.ins_manager = InstrumentManager(self.instrumentsBox, self.client,
                                             self.request_queue,
                                             self.queue_manager,
                                             self.touchOSCClient)
        self.ins_manager.addInstrument()

        # add clock...
        mgr = multiprocessing.Manager()
        self.clockVar = mgr.dict()
        self.clock = Clock(self.client, self.clockVar)

        self.BPM.trace('w', self.BPMchange)
        self.BPM.set(80)

        # add engine...
        #self.engine = Engine(self, self.ins_manager, self.BPM, touchOSCClient=self.touchOSCClient)
        self.engine = Engine2(self, self.client, self.clockVar, self.ins_manager)

        # add controls...
        self.controls = PlayerControls(self.maincontrols, self, self.engine)
        self.controls.grid(row=0, column=0, rowspan=2)

        # add status bar...
        self.statusBar = tk.Frame(self.mainframe, relief='sunken', bg=COLOR_SCHEME['dark_grey'], bd=2)
        self.statusLabel = tk.Label(self.statusBar, bg=self.statusBar.cget('bg'), fg=COLOR_SCHEME['text_light'],
            text='Connecting too: {}, Port:{}'.format(self.cl_ip, self.cl_port))
        self.statusLabel.bind('<Button-1>', self.editConnection)
        self.statusLabel.pack(side='right')

        # place everything...
        self.maincontrols.pack(padx=5, pady=5)
        self.instrumentsBox.pack(fill='both', expand=True, ipadx=4, ipady=4)
        self.statusBar.pack(side='bottom', fill='x')
        #self.maincontrols.grid(row=0, column=0, sticky='ew')
        #self.instrumentsBox.grid(row=1, column=0, sticky='nsew')
        #self.statusBar.grid(row=2, column=0, sticky='ew')

        # add OSC listeners...
        self.disp = dispatcher.Dispatcher()
        self.disp.map('/pingReply', self.pingReply, self.statusLabel)
        self.disp.map('/cc', self.ccRecieve)
        self.disp.map('/noteOn', self.noteOn)
        self.joystick_moved = False


        self.updateServer()
        self.listener = MessageListener(self.server)
        self.touchOSCListener = MessageListener(self.touchOSCServer)

        # start the loops...
        self.engine.daemon = True
        self.clock.start()
        self.engine.start()
        self.listener.start()
        self.touchOSCListener.start()
        self.checkConnection()
        #self.check_ins_updates()
        self.updateGUI()

        self.root.mainloop()

        # tidy up on close...
        print('Closing threads...')
        self.ins_manager.send_message('/panic', 0)
        self.clock.join(timeout=1)
        mgr.shutdown()
        self.engine.join(timeout=1)
        self.listener.join(timeout=1)
        self.touchOSCListener.join(timeout=1)
        self.network_manager.terminate()
        self.network_manager.join(timeout=1)

    def updateGUI(self):
        # only need to update if playing...
        bar = self.clockVar['bar']
        beat = self.clockVar['beat']
        text = '{:2}:{}'.format(bar, int(min(4, beat+1)))
        self.timeLabel.configure(text=text)

        # check if need to redraw canvases...
        for ins in self.ins_manager.get_instruments():
            if ins.request_GUI_update:
                ins.ins_panel.update_canvas()
                ins.request_GUI_update = False

            ins.ins_panel.move_canvas(beat)
            ins.ins_panel.update_buttons()
        self.controls.update_buttons()

        # 25 FPS refresh rate...
        self.root.after(1000//25, self.updateGUI)

    def checkConnection(self):
        try:
            self.client.send_message('/ping', 1)
        except:
            self.statusLabel.configure(text='NO INTERNET')
        else:
            self.statusLabel.configure(text='Checking connection to {}:{}'.format(
                                           self.sv_ip, self.sv_port))

        self.root.after(2000, self.checkConnection)

    def pingReply(self, msg, statusLabel, val):
        #print('connected!')
        statusLabel[0].configure(text='Sending to {}, Port {}'.format(self.cl_ip,
                                                                      self.cl_port))

    def editConnection(self, event):
        d = DialogOSCParams(self.root, title='Edit OSC connection...', baseApp=self)

    def updateClient(self):
        if self.client:
            del self.client
        self.client = udp_client.SimpleUDPClient(self.cl_ip, self.cl_port)
        print('Client updated: ', self.cl_ip, self.cl_port)

    def updateServer(self):
        if self.server:
            self.server.server_address = (self.sv_ip, self.sv_port)
        else:
            self.server = osc_server.BlockingOSCUDPServer((self.sv_ip, self.sv_port), self.disp)
        print('Server updated', self.sv_ip, self.sv_port)

    def parseOSC(self, addr, *args):
        items = addr.split('/')[1:]

        print('OSC Recieved:', (addr, args))
        print('   Items: ', items)
        print('   Args:  ', *args)

        if len(items) == 0:
            return

        if items[0] == 'armSelect':
            # arm selected instrument...
            pass
        elif items[0] == 'killAll':
            self.panic()
        elif items[0] == 'globalPlay':
            self.controls.play(None)
        elif items[0] == 'tempo':
            # deal with BPM change...
            bpm = self.bpmVar.get()
            if args[0] > 0.5:
                if bpm > 180:
                    return
                self.bpmVar.set(int(bpm) + 1)
            else:
                if bpm < 20:
                    return
                self.bpmVar.set(int(bpm) - 1)
        elif items[0] in '123456789':
            # instrument control
            if args[0] < 0.5:
                return
            _id = int(items[0]) - 1
            if _id not in self.ins_manager.instruments.keys():
                return

            ins = self.ins_manager.instruments[_id]

            if len(items) == 1:
                # set armed instrument?
                return

            ctrl = items[1]
            if ctrl == 'loopBars':
                if items[-1] == '1':
                    level = 0
                else:
                    level = 2**(int(items[-1])-2)

                ins.repeatSelect.set(level)
            elif ctrl == 'recBars':
                if items[-1] == '1':
                    level = 0
                else:
                    level = 2**(int(items[-1])-2)

                ins.recSelect.set(level)
            elif ctrl == 'loopRhythm':
                if items[-1] == '1':
                    level = 0
                else:
                    level = int(items[-1])-1

                ins.rhythmSelect.set(level)
            elif ctrl == 'hold':
                ins.toggle_hold()

            elif ctrl == 'mute':
                ins.toggle_mute()

            elif ctrl in META_PARAMS:
                ins.updateMetaParams({ctrl: args[0]})

        else:
            print('Unknown OSC message...')


    def ccRecieve(self, *msg):
        print('CC Recieve:', msg)
        val = msg[1]
        num = msg[2]
        # here process CC messages (value, number)
        CC_FUNCS = {1:  self.CC_select_ins,
                    9:  self.CC_play,
                    10: self.CC_rec,
                    11: self.CC_stop,
                    12: self.CC_add,
                    17: self.CC_arm_selected,
                    18: self.CC_arm_selected,
                    19: self.CC_arm_selected,
                    20: self.CC_arm_selected,
                    21: self.CC_loop_selected,
                    22: self.CC_loop_selected,
                    23: self.CC_loop_selected,
                    24: self.CC_loop_selected,
                    25: self.CC_change_parameter,
                    26: self.CC_change_parameter,
                    27: self.CC_change_parameter,
                    28: self.CC_change_parameter,
                    29: self.CC_change_parameter,
                    30: self.CC_change_parameter,
                    31: self.CC_change_parameter,
                    #32: self.CC_change_parameter
                   }

        # since MIDI TOGGLE sends two separate messages, only listen to
        # non-zero messages (a little sketchy...)
        if num < 25 and val == 0:
            return
        else:
            try:
                func = CC_FUNCS[num]
            except:
                print('Unassigned CC function')
                return

            func(val, num)

    def CC_play(self, val, num):
        #self.toggle_playback()
        self.controls.play(None)

    def CC_rec(self, val, num):
        #self.toggle_record()
        self.controls.record(None)

    def CC_stop(self, val, num):
        self.stop()

    def CC_add(self, val, num):
        self.ins_manager.addInstrument()

    def CC_select_ins(self, val, num):
        if abs(val - 64) > 32:
            if not self.joystick_moved:
                # check if moved up or down
                if val < 64:
                    self.ins_manager.select_down()
                else:
                    self.ins_manager.select_up()
            self.joystick_moved = True

        else:
            self.joystick_moved = False


    def CC_arm_selected(self, val, num):
        VALS = {17: 1, 18: 2, 19: 4, 20: 8}
        selected_ins = self.ins_manager.selected_ins
        n = VALS[num]
        if len(selected_ins.record_bars) > 0:
            if selected_ins.record_bars[-1] - selected_ins.record_bars[0] == n:
                n = 0
        self.ins_manager.set_recording_instrument(selected_ins, n)

    def CC_loop_selected(self, val, num):
        VALS = {21: 1, 22: 2, 23: 4, 24: 8}
        n = VALS[num]
        try:
            selected_ins = self.ins_manager.selected_ins.ins_id
        except:
            print('No selected instrument to set loop')
            return

        if self.ins_manager.selected_ins.loopLevel == n:
            n = 0

        self.ins_manager.instrumentPanels[selected_ins].repeatSelect.set(n)

    def CC_change_parameter(self, val, num):
        p = val/127

        VALS = {25: 'span',
                26: 'tCent',
                27: 'cDens',
                28: 'cDepth',
                29: 'jump',
                30: 'rDens',
                31: 'pos'}

        try:
            selected_ins = self.ins_manager.selected_ins.ins_id
        except:
            print('No selected instrument to update')
            return

        param = VALS[num]
        self.ins_manager.instrumentPanels[selected_ins].changeParameter(param, p)

    def noteOn(self, *msg):
        # MIDI noteOn event
        if self.ins_manager.armed_ins:
            print('armed ins accepts note', msg)
            self.ins_manager.armed_ins.noteOn = msg[1]
        print('NoteOn: {}'.format(msg))

    def stop(self):
        #self.engine.toggle_stop()
        if not self.clockVar['playing']:
            for ins in self.ins_manager.get_instruments():
                ins.reset()
        self.clock.toggle_stop()

    def panic(self):
        '''Send MIDI all off message to all channels'''
        self.ins_manager.send_message('/panic', 0)

    def sendCC(self, num, val):
        self.client.send_message('/midiCC', (num, val))

    def BPMchange(self, *args):
        #print('BPM change: ', args)
        self.clockVar['bpm'] = self.BPM.get()
        if self.touchOSCClient:
            self.touchOSCClient.send_message('/bpm', self.BPM.get())


class DialogOSCParams(Dialog):
    def body(self, root):
        try:
            self.baseApp = self.kwargs['baseApp']
        except:
            print('Error: BaseApp not supplied')
            self.cancel()

        tk.Label(root, text='Client Address:').grid(row=0, columnspan=2)
        tk.Label(root, text='IP:').grid(row=1)
        tk.Label(root, text='Port:').grid(row=2)

        tk.Label(root, text='Server Address:').grid(row=0, column=2, columnspan=2)
        tk.Label(root, text='IP:').grid(row=1, column=2)
        tk.Label(root, text='Port:').grid(row=2, column=2)

        self.cl_ip = tk.Entry(root)
        self.cl_ip.insert(0, self.baseApp.cl_ip)
        self.cl_port = tk.Entry(root)
        self.cl_port.insert(0, self.baseApp.cl_port)
        self.sv_ip = tk.Entry(root)
        self.sv_ip.insert(0, self.baseApp.sv_ip)
        self.sv_port = tk.Entry(root)
        self.sv_port.insert(0, self.baseApp.sv_port)

        self.cl_ip.grid(row=1, column=1)
        self.cl_port.grid(row=2, column=1)
        self.sv_ip.grid(row=1, column=3)
        self.sv_port.grid(row=2, column=3)

        return self.cl_ip

    def validate(self):
        return 1

    def apply(self):
        self.baseApp.cl_ip = str(self.cl_ip.get())
        self.baseApp.cl_port = int(self.cl_port.get())
        self.baseApp.updateClient()

        self.baseApp.sv_ip = str(self.sv_ip.get())
        self.baseApp.sv_port = int(self.sv_port.get())
        self.baseApp.updateServer()
        print('Connection updated')


if __name__ == '__main__':
    #multiprocessing.freeze_support()   # needed to build for Windows
    print('Starting {}'.format(APP_NAME))

    app = MusaicApp()

# EOF

