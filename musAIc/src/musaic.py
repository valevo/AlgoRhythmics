import tkinter as tk
from tkinter import font
from tkinter.ttk import Style
import threading
import multiprocessing
import queue
import time
from collections import deque

from guielements import VScrollFrame, InstrumentPanel, PlayerControls, Knob
import utils
from simpleDialog import Dialog
from networkEngine import NetworkManager
from pythonosc import udp_client, osc_server, dispatcher

import numpy as np

# Address and port of host machine to send messages to OSC-MIDI bridge:
# ON WINDOWS:
#   - run ipconfig in command prompt for Address (Wireless Adapter WiFi)
#   - run NetAddr.localAddr; in SuperCollider to get port number
#CLIENT_ADDR = '192.168.1.12'
#CLIENT_ADDR = '192.168.0.36'
#CLIENT_ADDR = '145.109.5.179'
CLIENT_ADDR = '100.75.0.230'
#CLIENT_ADDR = '127.0.0.1'
CLIENT_PORT = 57120

# Address and port of guest machine to listen for incoming messages:
#   - run ifconfig (or `ip addr` in Arch) and use Host-Only IP (same as listed on Windows ipconfig)
#   - port can be arbitrary, but SuperCollider must know!
SERVER_ADDR = '192.168.56.102'
#SERVER_ADDR = '192.168.56.102'
#SERVER_ADDR = '127.0.0.1'
SERVER_PORT = 7121

APP_NAME = 'musAIc LIVE (v0.4)'


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

                    self.ins_manager.set_playing()

                # before bar processes
                for ins in self.ins_manager.get_instruments():
                    ins.load_bar()
                    ins.ins_panel.update_canvas()

                try:
                    new_bps = self.bpmVar.get()/60
                    bps = new_bps
                except:
                    pass
                while self.beat < 4 and not self.stop_request.isSet():
                    # during bar processes...
                    self.wait_event.wait(timeout=0.05)
                    self.beat = min(4, bps*(time.time() - time_bar_start))
                    self.ins_manager.play_instruments(self.bar, self.beat)
                    for ins in self.ins_manager.instrumentPanels.values():
                        ins.move_canvas(self.beat)

                # after bar processes
                for ins in self.ins_manager.get_instruments():
                    ins.close_bar()
                    ins.increment_bar_num()

                #self.ins_manager.load_nextbar()

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
                        print('resetting positions')
                        self.stop = False
                        for ins in self.ins_manager.get_instruments():
                            ins.reset()
                        self.bar = 0
                        self.ins_manager.load_nextbar()

                    self.app.controls.update_buttons()
                    self.start_playing = True
                    self.ins_manager.set_stopped()
                    self.ins_manager.update_gui()

            # update instruments if they have requested new bars...
            for ins in self.ins_manager.get_instruments():
                ins.check_updates()

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

    def toggle_stop(self, stop_button=None):
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
        self.ins_panel = None
        self.request_queue = request_queue   # queue to send new bar requests
        self.network_queue = network_queue   # queue to recieve new bars

        self.transpose = 0                # number of octaves to shift output
        self.confidence = 0               # the rank of which note to play
        self.bar_num = 0                  # current bar number
        self.stream = utils.Stream()      # the musical stream of notes / chords
        self.status = PAUSED              # current playing status
        self.armed = False                # is armed (recording) instrument
        self.active = True                # is selected instrument
        self.mute = False                 # mute the output
        self.continuous = True            # continuously generate new bars
        self.recording = False            # if instrument is recording input
        self.record_bars = []             # range of recording bars [start, end]

        self.lead = False                 # Lead instrument that others follow

        self.current_bar = None           # currently playing bar
        #self.lastNote = []                # last note played for muting
        self.noteOffEvents = dict()       # a queue of noteOff events: (time, note)
        self.offMode = 0                  # note off mode: 0 - hold to noteOn
                                          #                1 - hold for exactly one beat

        self.loopLevel = 0                # number of bars to loop 
        self.loopEnd = 0                  # bar the loop number offsets from
        self.rhythmLoopLevel = 0          # what level to loop the rhythm information in generation

        # SHOULD BE QUEUE?
        self.noteOn = None                # any noteOn events placed here
        self.recorded_bar = {}            # currently recorded bar

        # request some bars
        for _ in range(4):
            self.request_new_bar()

    def set_ins_panel(self, ins_panel):
        self.ins_panel = ins_panel

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
        print('LOAD BAR', self.bar_num)
        if self.status == PAUSED or self.status == PAUSE_WAIT:
            return

        if self.status == PLAY_WAIT:
            self.status = PLAYING

        if self.continuous:
            for _ in range(max(0, -len(self.stream) + self.bar_num + 6)):
                self.request_new_bar()

        if self.loopLevel > 0:
            if self.bar_num > self.loopEnd:
                self.bar_num = self.loopEnd - self.loopLevel + 1
                self.bar_num = max(0, self.bar_num)

                # reset played attribute of notes...
                for i in range(self.loopLevel):
                    bar = self.stream.getBar(self.bar_num + i)
                    for n in bar.values():
                        n.played = False

        if self.record_bars:
            if self.bar_num >= self.record_bars[0] and \
               self.bar_num < self.record_bars[1]:
                self.recording = True
        else:
            self.recording = False
            try:
                self.current_bar = self.stream.getBar(self.bar_num)

            except Exception as e:
                # no bar left to load...
                print(e)
                self.current_bar = None

        if self.ins_panel:
            self.ins_panel.update_canvas()

    def play_bar(self, bar, beat):
        '''Checks if have to play any notes in current bar'''
        #print('PLAY BAR', self.bar_num, bar, beat)
        # check if any new bars are ready...
        self.check_updates()
        #print(bar, beat)

        #if self.current_bar == None and len(self.bars) > 0:
        #    self.current_bar = self.bars_queue.get(block=False)

        #print(self.status, self.mute, self.current_bar)
        if self.status == PAUSED or self.mute:# or self.current_bar == None:
            return

        if self.recording:
            #print('REC: ', bar, beat)
            # parse any inputs recieved, something like...
            if self.noteOn:
                print('play_bar: noteOn', self.noteOn)
                self.recorded_bar[round(beat*8)/8] = self.noteOn
                self.noteOn = None
            return

        current_note = self.stream.getNotePlaying(self.bar_num, beat)

        #self.check_note_off(beat)

        if current_note:
            if current_note.midi < 0:
                # rest...
                return

            if not current_note.played:
                self.mute_last_notes()
                note = current_note.midi + 12*self.transpose
                vel = np.random.randint(70, 90)
                if current_note.isChord():
                    self.lastNote = []
                    for c in current_note.chord:
                        cNote = note + c
                        self.ins_manager.send_message('/noteOn', (self.chan, cNote, vel))
                        #print(self.bar_num)
                        #print('Play note {} on channel {}'.format(cNote, self.chan))
                        self.lastNote.append(cNote)
                        #self.noteOffStack.append()

                else:
                    self.ins_manager.send_message('/noteOn', (self.chan, note, vel))
                    #print(self.bar_num)
                    #print('Play note {} on channel {}'.format(note, self.chan))
                    self.lastNote = [note]

                current_note.played = True

    def check_note_off(self, beat):
        if self.offMode == 0:
            # noteOff on next noteOn
            pass
        elif self.offMode == 1:
            # noteOff after one beat 
            keyList = []
            for n, t in self.noteOffEvents.items():
                if t < self.bar_num + beat:
                    keyList.append(n)
                    self.ins_manager.send_message('/noteOff', (self.chan, n.))



    def mute_last_notes(self, beat):
        for n in self.lastNote:
            self.ins_manager.send_message('/noteOff', (self.chan, n))

    def mute_all_notes(self):
        self.ins_manager.send_message('/allOff', self.chan)

    def request_new_bar(self):
        msg = {
            'confidence': self.confidence,
            'loopRhythm': self.rhythmLoopLevel,
        }
        self.request_queue.put((1, self.ins_id, msg))

    def check_updates(self):
        '''Checks if any news from network'''
        if self.recording or self.armed:
            return
        try:
            while not self.network_queue.empty():
                msg = self.network_queue.get(block=False)

                if 'bar' in msg:
                    new_bar = msg['bar']

                    self.stream.appendBar(new_bar)
                    if self.ins_panel:
                        self.ins_panel.update_canvas()

                if 'md' in msg:
                    self.ins_panel.updateMetaParams(msg['md'])

        except Exception as e:
            # no new bars...
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
                self.ins_panel.update_canvas()
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

    def update_params(self, parameters):
        self.request_queue.put((3, self.ins_id, parameters))

    def toggle_paused(self):
        if self.status == PAUSED or self.status == PAUSE_WAIT:
            self.status = PLAY_WAIT
        else:
            self.status = PAUSE_WAIT

    def toggle_loop(self, level):
        if self.loopLevel == level:
            self.loopLevel = 0
        else:
            self.loopLevel = level
            self.loopEnd = self.bar_num

        print(self.loopLevel, self.loopEnd)

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


class InstrumentManager():
    """
    Holds all the instruments and manages their processes
    Midi player and GUI threads use this
    """
    def __init__(self, ins_box, client, request_queue, queue_manager):
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
        for _id in self.instruments.keys():
            self.instruments[_id].load_bar()

    def play_instruments(self, bar, beat):
        for ins in self.get_instruments():
            if ins.status == PLAYING or ins.status == PAUSE_WAIT:
                ins.play_bar(bar, beat)

    def get_instruments(self):
        return list(self.instruments.values())

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
        for ins in self.instruments.values():
            ins.status = PAUSED

    def update_gui(self):
        for ins_panel in self.instrumentPanels.values():
            ins_panel.move_canvas(0)

    def set_lead_instrument(self, ins):
        if self.lead_ins:
            self.lead_ins.lead = False

        self.lead_ins = ins
        if ins:
            self.lead_ins.lead = True

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

    def send_message(self, add, msg):
        try:
            self.client.send_message(add, msg)
        except OSError:
            print('[OS Error] Instrument manager cannot send message {}:{}'.format(
                      add, msg))
            return


class MusaicApp():
    client = None
    server = None

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_NAME)
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

        self.queue_manager = multiprocessing.Manager()
        self.request_queue = self.queue_manager.Queue()
        self.network_manager = NetworkManager(self.request_queue)

        self.network_manager.start()

        self.mainframe = tk.Frame(self.root)
        self.mainframe['bg'] = '#101010'
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
        self.BPM.set(80)
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
            text='Connecting too: {}, Port:{}'.format(self.cl_ip, self.cl_port))
        self.statusLabel.bind('<Button-1>', self.editConnection)
        self.statusLabel.pack(side='right')
        self.statusBar.pack(side='bottom', fill='x')

        # add OSC listener...
        self.disp = dispatcher.Dispatcher()
        self.disp.map('/pingReply', self.pingReply, self.statusLabel)
        self.disp.map('/cc', self.ccRecieve)
        self.disp.map('/noteOn', self.noteOn)
        self.joystick_moved = False

        #self.server = osc_server.BlockingOSCUDPServer((self.sv_ip, self.sv_port), self.disp)
        self.updateServer()
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
        self.ins_manager.send_message('/panic', 0)
        self.engine.join(timeout=1)
        self.listener.join(timeout=1)
        self.network_manager.terminate()
        self.network_manager.join(timeout=1)


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

        #for ins in self.ins_manager.instrumentPanels.values():
        #    ins.move_canvas(beat)
            # set record buttons arcordingly

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
        items = addr.split('/')

        print('OSC Recieved:', (add, args))
        print('   Items: ', items)

        if len(items) == 0:
            return

        if items[0] == 'armSelect':
            # arm selected instrument...
            pass
        elif items[0] == 'killAll':
            self.panic()
        elif items[0] == 'globalPlay':
            self.controls.play()
        elif items[0] == 'bpm':
            # deal with BPM change...
            pass
        elif items[0] in range(8):
            # instrument control
            _id = items[0]
            ctrl = items[1]

            if ctrl in 

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
                    #31: self.CC_change_parameter,
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
                30: 'rDens'}

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
        self.engine.toggle_stop()

    def panic(self):
        '''Send MIDI all off message to all channels'''
        self.ins_manager.send_message('/panic', 0)

    def sendCC(self, num, val):
        self.client.send_message('/midiCC', (num, val))

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

