import tkinter as tk
from tkinter import font
import threading
import multiprocessing
import queue
import time
from collections import deque

from guielements import VScrollFrame, InstrumentPanel, PlayerControls
from networkEngine import Network
from pythonosc import udp_client, osc_server, dispatcher

import numpy as np

# Address and port of host machine to send messages to OSC-MIDI bridge:
# ON WINDOWS:
#   - run ipconfig in command prompt for Address (Wireless Adapter WiFi)
#   - run NetAddr.localAddr; in SuperCollider to get port number
#CLIENT_ADDR = '100.75.0.230'
#CLIENT_PORT = 57120
CLIENT_ADDR = '145.109.6.217'
CLIENT_PORT = 57120

# Address and port of guest machine to listen for incoming messages:
#   - run ifconfig and use Host-Only IP (same as listed on Windows ipconfig)
#   - port can be arbitrary, but SuperCollider must know!
SERVER_ADDR = '192.168.56.103'
SERVER_PORT = 7121

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


class Engine(threading.Thread):
    """Controls the clock and instument playback"""
    def __init__(self, app, ins_manager, bpmVar):
        threading.Thread.__init__(self)
        self.app = app
        self.ins_manager = ins_manager
        self.bpmVar = bpmVar

        self.stop_request = threading.Event()
        self.play_request = threading.Event()
        self.is_playing = threading.Event()
        self.wait_event = threading.Event()
        self.beat = 0
        self.bar = 0
        self.start_playing = True
        self.record = False
        self.stop = False

    def run(self):
        while not self.stop_request.isSet():
            if self.play_request.isSet():
                if self.start_playing:
                    # start bar process...
                    self.is_playing.set()
                    time_bar_start = time.time()
                    self.start_playing = False
                    if self.record:
                        self.app.sendCC(1, 127)


                bps = self.bpmVar.get()/60
                while self.beat < 4 and not self.stop_request.isSet():
                    # during bar processes...
                    self.wait_event.wait(timeout=0.01)
                    self.beat = min(4, bps*(time.time() - time_bar_start))
                    self.ins_manager.play_instruments(self.bar, self.beat)

                # after bar processes
                for ins in self.ins_manager.get_instruments():
                    ins.close_bar()

                self.ins_manager.load_nextbar()

                self.bar += 1
                self.beat = 0.0
                time_bar_start += 4/bps
            else:
                # after stopping playback...
                self.is_playing.clear()
                if not self.start_playing:
                    self.app.sendCC(1, 0)
                    self.app.sendCC(2, 127)
                    self.record = False

                    if self.stop:
                        print('reseting positions')
                        self.stop = False
                        for ins in self.ins_manager.get_instruments():
                            ins.bar_num = -1
                        self.bar = 0
                        self.ins_manager.load_nextbar()

                    self.app.controls.update_buttons()
                    self.start_playing = True

            # update instruments if they have requested new bars...
            for ins in self.ins_manager.get_instruments():
                ins.check_new_bar()

            self.play_request.wait(timeout=0.1)

    def join(self, timeout=None):
        self.stop_request.set()
        self.wait_event.set()
        self.play_request.clear()
        super(Engine, self).join(timeout)

    def toggle_playback(self):
        if self.play_request.isSet():
            self.play_request.clear()
        else:
            self.play_request.set()

    def toggle_stop(self, stop_button):
        if not self.stop:
            self.stop = True
            self.play_request.clear()
        else:
            self.stop = False
            self.play_request.set()


PAUSED = 2
PLAY_WAIT = -1
PAUSE_WAIT = -2
PLAYING = 1


class Instrument():
    def __init__(self, chan, ins_id, ins_manager, request_queue, network_queue):
        self.chan = chan
        self.ins_id = ins_id
        self.ins_manager = ins_manager
        self.request_queue = request_queue   # queue to send new bar requests
        self.network_queue = network_queue   # queue to recieve new bars

        self.transpose = 0                # number of octaves to shift output
        self.confidence = 0               # the rank of which note to play
        self.bar_num = 0                  # current bar number
        self.bars = []                    # list of all bars 
        self.status = PLAYING             # current playing status
        self.armed = False                # is armed (recording) instrument
        self.active = True                # is selected instrument
        self.mute = False                 # mute the output
        self.continuous = True            # continuously generate new bars
        self.recording = False            # if instrument is recording input
        self.record_bars = []             # list of bars that will take input

        self.current_bar = None           # currently playing bar
        self.eventTimes = deque([])       # queue of event times in this bar
        self.lastNote = None              # last note played for muting

        self.loopLevel = 0                # number of bars to loop 
        self.loopEnd = 0                  # bar the loop number offsets from

        # SHOULD BE QUEUE?
        self.noteOn = None                # any noteOn events placed here
        self.recorded_bar = {}            # currently recorded bar

        # request some bars
        for _ in range(4):
            self.request_new_bar()


    def delete(self):
        '''Deletes this instrument'''
        self.ins_manager.removeInstrument(self.ins_id)

    def load_bar(self):
        '''Loads the current bar into memory to be played'''
        if self.status == PAUSED or self.status == PAUSE_WAIT:
            return

        if self.status == PLAY_WAIT:
            self.status = PLAYING

        if self.continuous:
            for _ in range(max(0, -len(self.bars) + self.bar_num + 6)):
                self.request_new_bar()

        if self.loopLevel > 0:
            if self.bar_num == self.loopEnd:
                self.bar_num -= self.loopLevel
                self.bar_num = max(0, self.bar_num)

        try:
            self.current_bar = self.bars[self.bar_num]
            self.eventTimes = deque(sorted(list(self.current_bar.keys())))

            if self.bar_num in self.record_bars:
                self.recording = True
            else:
                self.recording = False
                self.record_bars = []

            self.bar_num += 1
        except:
            # no bar left to load...
            self.current_bar = None
            #self.status = PAUSE_WAIT
            return

    def play_bar(self, bar, beat):
        '''Checks if have to play any notes in current bar'''
        # check if any new bars are ready...
        self.check_new_bar()

        #if self.current_bar == None and len(self.bars) > 0:
        #    self.current_bar = self.bars_queue.get(block=False)

        if self.status == PAUSED or self.mute or self.current_bar == None:
            return

        if len(self.eventTimes) == 0:
            return

        if self.recording:
            # parse any inputs recieved, something like...
            if self.noteOn:
                self.recorded_bar[round(beat*4)/4] = self.noteOn
                self.noteOn = None

            return

        if beat >= self.eventTimes[0]:
            # play note
            time = self.eventTimes.popleft()
            note = self.current_bar[time] + 12 * self.transpose

            if self.lastNote:
                self.ins_manager.send_message('/noteOff',
                                                     (self.chan, self.lastNote))
            self.ins_manager.send_message('/noteOn', (self.chan, note))
            #print('Play note {} on channel {}'.format(note, self.chan))
            self.lastNote = note

    def request_new_bar(self):
        self.request_queue.put((1, self.ins_id, self.confidence))

    def check_new_bar(self):
        '''Checks if a new bar from network is ready'''
        try:
            bar = self.network_queue.get(block=False)
            self.bars.append(bar)
        except:
            # no new bars...
            return

    def close_bar(self):
        '''Called at end of bar'''
        if self.lastNote:
            self.ins_manager.send_message('/noteOff', (self.chan,
                                                       self.lastNote))

        if self.status == PAUSE_WAIT:
            self.status = PAUSED

        if self.bar_num == len(self.bars) - 1 and not self.continuous:
            self.status = PAUSED

        if self.recording:
            # only update bar if anything was recorded
            if len(self.recorded_bar) > 0:
                self.bars[self.bar_num - 1] = dict(self.recorded_bar)
                self.recorded_bar = {}

            self.recording = False

    def init_record(self, num):
        if num == 0:
            self.record_bars = []
            self.armed = False
        else:
            start = self.bar_num
            self.record_bars = list(range(start, start+num))

            # set up loop for when finished...
            self.loopLevel = len(self.record_bars)
            self.loopEnd = self.record_bars[0]


    def toggle_paused(self):
        if self.status == PAUSED or self.status == PAUSE_WAIT:
            self.status = PLAY_WAIT
        else:
            self.status = PAUSE_WAIT

    def toggle_mute(self):
        self.mute = not self.mute

    def toggle_continuous(self):
        self.continuous = not self.continuous


class InstrumentManager():
    """
    Holds all the instruments and manages their processes
    Midi player and GUI threads use this
    """
    def __init__(self, ins_box, client, request_queue, queue_manager):
        self.instruments = {}
        self.instrumentPanels = {}
        self.ins_box = ins_box
        self.client = client
        self.request_queue = request_queue
        self.queue_manager = queue_manager
        self.ins_counter = 0
        self.armed_ins = None
        self.selected_ins = None

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
        self.instrumentPanels[self.ins_counter] = insPanel
        self.instruments[self.ins_counter] = instrument
        self.ins_counter += 1
        insPanel.pack(side='bottom', fill='x', pady=5)
        self.set_selected_instrument(instrument)

    def removeInstrument(self, ins_id):
        if self.selected_ins.ins_id == ins_id:
            try:
                next_id = list(self.instruments.keys())[0]
                self.set_selected_instrument(self.instruments[next_id])
            except:
                self.selected_ins = None

        del self.instruments[ins_id]
        self.instrumentPanels[ins_id].destroy()
        del self.instrumentPanels[ins_id]
        self.request_queue.put((-1, ins_id))
        print('Removing instrument {}'.format(ins_id))

    def load_nextbar(self):
        for ins in self.get_instruments():
            ins.load_bar()

    def play_instruments(self, bar, beat):
        for ins in self.get_instruments():
            if ins.status == PLAYING or ins.status == PAUSE_WAIT:
                ins.play_bar(bar, beat)


    def get_instruments(self):
        return list(self.instruments.values())

    def set_recording_instrument(self, ins, num):
        if self.armed_ins:
            self.armed_ins.init_record(0)
            self.instrumentPanels[self.armed_ins.ins_id].recVar.set(0)

        if num == 0:
            self.armed_ins = None
        else:
            self.armed_ins = ins
            self.armed_ins.init_record(num)
            self.instrumentPanels[self.armed_ins.ins_id].recVar.set(num)

    def set_selected_instrument(self, ins):
        if self.selected_ins:
            self.selected_ins.active = False

        self.selected_ins = ins
        if ins:
            self.selected_ins.active = True

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

    def send_message(self, add, msg):
        try:
            self.client.send_message(add, msg)
        except OSError:
            return

APP_NAME = 'musAIc LIVE (v0.3)'

class MusaicApp():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_NAME)
        hs = self.root.winfo_screenheight()
        self.root.geometry('1000x600+0+%d'%(hs/2))
        self.root.resizable(0, 0)

        self.client = udp_client.SimpleUDPClient(CLIENT_ADDR, CLIENT_PORT)

        self.queue_manager = multiprocessing.Manager()
        self.request_queue = self.queue_manager.Queue()
        self.network = Network(self.request_queue)

        self.network.start()

        self.mainframe = tk.Frame(self.root)
        self.mainframe.pack_propagate(False)
        self.mainframe.pack(fill='both', expand=True)
        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.rowconfigure(0, weight=1)

        # main control panel...
        self.maincontrols = tk.Frame(self.mainframe)

        timeFont = font.Font(family='Courier', size=12, weight='bold')
        self.timeLabel = tk.Label(self.maincontrols, text='00:00', bg='#303030', fg='yellow',
                             width=10, relief='sunken', bd=2,
                             font='Arial 16', padx=5, pady=5)

        self.BPM = tk.IntVar(self.maincontrols)
        self.BPM.set(120)
        self.bpmBox = tk.Spinbox(self.maincontrols, from_=20, to=240, textvariable=self.BPM,
                            width=3)

        self.panicButton = tk.Button(self.maincontrols, text='Panic',
                                     command=self.panic)
        # pack main controls...
        self.timeLabel.grid(row=0, column=4, rowspan=2, padx=5)
        tk.Label(self.maincontrols, text='Tempo:').grid(row=0, column=5, sticky='e')
        self.bpmBox.grid(row=0, column=6)
        self.panicButton.grid(row=1, column=5, columnspan=2, sticky='ew')

        # add instrument button and panels...

        self.maincontrols.pack(padx=5, pady=5)

        self.instrumentsBox = tk.Frame(self.mainframe, relief='sunken', bd=2)
        self.instrumentsBox.pack(fill='both', expand=True, ipadx=4, ipady=4)

        self.ins_manager = InstrumentManager(self.instrumentsBox, self.client,
                                             self.request_queue,
                                             self.queue_manager)
        self.ins_manager.addInstrument()

        # add engine...
        self.engine = Engine(self, self.ins_manager, self.BPM)

        # add controls...
        self.controls = PlayerControls(self.maincontrols, self.engine)
        self.controls.grid(row=0, column=0, rowspan=2)

        # add status bar
        self.statusBar = tk.Frame(self.mainframe, relief='sunken', bd=2)
        self.statusLabel = tk.Label(self.statusBar, 
                                    text='Connecting too: {}, Port: {}'.format(CLIENT_ADDR, CLIENT_PORT))
        self.statusLabel.pack(side='right')
        self.statusBar.pack(side='bottom', fill='x')

        # add OSC listener...
        self.disp = dispatcher.Dispatcher()
        self.disp.map('/pingReply', self.pingReply, self.statusLabel)
        self.disp.map('/cc', self.ccRecieve)
        self.disp.map('/noteOn', self.noteOn)
        self.joystick_moved = False

        self.server = osc_server.BlockingOSCUDPServer((SERVER_ADDR, SERVER_PORT), self.disp)
        self.listener = MessageListener(self.server)

        # start the loops...
        self.engine.daemon = True
        self.engine.start()
        self.listener.start()
        self.checkConnection()
        self.updateGUI()
        self.root.mainloop()

        # tidy up on close...
        print('Closing threads...')
        self.ins_manager.send_message('/allOff', 0)
        self.engine.join(timeout=1)
        self.listener.join(timeout=1)
        self.network.terminate()
        self.network.join(timeout=1)


    def updateGUI(self):
        # only need to update if playing...
        bar = self.engine.bar
        beat = self.engine.beat
        text = '{:2}:{}'.format(bar, int(min(4, beat+1)))
        self.timeLabel.configure(text=text)

        #if self.engine.record:
        #    self.recButton['relief'] = 'sunken'
        #    self.recButton['fg'] = 'red'
        #else:
        #    self.recButton['relief'] = 'raised'
        #    self.recButton['fg'] = 'black'

        for ins in self.ins_manager.instrumentPanels.values():
            ins.update_display(beat)
            # set record buttons arcordingly

        # 25 FPS refresh rate...
        self.root.after(40, self.updateGUI)

    def checkConnection(self):
        try:
            self.client.send_message('/ping', 1)
        except:
            self.statusLabel.configure(text='NO INTERNET')
        else:
            self.statusLabel.configure(text='Checking connection...')

        self.root.after(2000, self.checkConnection)

    def pingReply(self, msg, statusLabel, val):
        #print('connected!')
        statusLabel[0].configure(text='Sending to {}, Port {}'.format(CLIENT_ADDR,
                                                                      CLIENT_PORT))

    def ccRecieve(self, *msg):
        print(msg)
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
                    24: self.CC_loop_selected
                   }

        # since MIDI TOGGLE sends two separate messages, only listen to
        # non-zero messages (a little sketchy...)
        if val == 0:
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
        if len(selected_ins.record_bars) == n:
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


    def noteOn(self, *msg):
        # MIDI noteOn event
        if self.ins_manager.armed_ins:
            self.ins_manager.armed_ins.noteOn = msg[1]
        print('NoteOn: {}'.format(msg))

    def stop(self):
        self.engine.toggle_stop()

    def panic(self):
        '''Send MIDI all off message to all channels'''
        self.ins_manager.send_message('/allOff', 0)

    def sendCC(self, num, val):
        self.client.send_message('/midiCC', (num, val))



if __name__ == '__main__':
    print('Starting {}'.format(APP_NAME))

    app = MusaicApp()

# EOF

