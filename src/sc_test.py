import time, random
import threading
from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server

BPM = 120
IOI = 60/BPM
divisions = [1, 0.5, 0.25, 0.75]
major = [0, 2, 4, 5, 7, 9, 11]
major_pen = [0, 2, 4, 7, 9]
root = 60

global spread
spread = 2

def construct_note_range(scale):
    nr = []
    for i in range(-2, 2):
        for deg in scale:
            nr.append(root + i*12 + deg)

    nr.append(root+24)
    return nr

def generate_rhythm():
    bars = random.randint(1, 5)

    rhy = []
    while True:
        r = random.choice(divisions)
        if sum(rhy) + r <= bars:
            rhy.append(r)
        elif sum(rhy) == bars:
            return rhy

def change_spread(*msg):
    global spread
    spread = int(msg[1])
    print(spread)

class musicController(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        server.serve_forever()

class musicBot(threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID

    def run(self):
        print('Start...')
        notes = construct_note_range(major_pen)
        degree = len(notes)//2

        # start synth
        client.send_message("/setup_synth", 0)

        while True:
            rhythm = generate_rhythm()

            client.send_message("/set_trem", random.randint(1, 6))

            for i in range(random.randint(2, 5)):
                for r in rhythm:
                    beat = IOI * random.choice(divisions)
                    degree += random.choice(range(-spread, spread+1))

                    degree = min(len(notes)-1, max(0, degree))

                    client.send_message("/play_note", notes[degree])
                    time.sleep(beat)

            print('Next rhythm...')

disp = dispatcher.Dispatcher()
disp.map('/spread', change_spread)

client = udp_client.SimpleUDPClient('145.107.37.248', 57120)
#server = osc_server.BlockingOSCUDPServer(('127.0.0.1', 7101), disp)


bot = musicBot(1)
listerner = musicController()
bot.start()
#listerner.start()
#bot.join()
#listerner.join()
