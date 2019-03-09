

def parseBarData(notes, octaves, rhythm, chords=None, ts_num=4):
    ''' Converts the separate melody, rhythm and chord data into a new bar
    and appends it to the current stream'''

    if len(rhythm) != ts_num:
        print('Wrong time signature for rhythm, attempting to fix...')
        print(f'(Expected {ts_num}, got {len(rhythm)})')
        ts_num = len(rhythm)

    RES = len(notes)
    beat_slot = RES/ts_num

    r_mask = [False]*RES
    for i, beat in enumerate(rhythm):
        for j in beat:
            index = round((i + j%1)*beat_slot)
            if index >= RES:
                continue
            r_mask[index] = True

    bar = {}
    for i in range(RES):
        if r_mask[i]:
            offset = i*ts_num/RES
            pc = notes[i]
            if pc == None:
                # rest...
                bar[offset] = -1
            else:
                o = octaves[i]
                bar[offset] = PCOctaveToMIDI(int(pc), o)

    return bar


def PCOctaveToMIDI(pc, octave):
    if pc > 12:
        # chord...
        pc -= 12
    return 12*(octave+1) + pc - 1

def MIDItoPCOctave(midi):
    return (midi%12+1, midi//12-1)

def convertStreamToData(stream):
    ''' Converts a dictionary of input data to separate rhythm, melody, chords  '''
    RES = 48

    rhythm = []
    melody = []
    octave = []
    chords = []

    for bar in stream.getBars():
        b_melody = [None]*RES
        b_octave = [None]*RES
        b_chords = []

        beats = [[] for i in range(stream.ts_num)]

        for b in bar.keys():
            try:
                beats[int(b)].append(b%1)
                idx = int(RES/stream.ts_num * b)
                note = bar[b]
                pc, oc = MIDItoPCOctave(note.midi)
                if note.isChord():
                    b_chords.append(note.chord)
                    pc += 12
                b_melody[idx] = pc
                b_octave[idx] = oc

            except IndexError:
                print('Beat outside scope of time signature')
                continue

        rhythm.append(tuple(map(tuple, beats)))
        melody.append(tuple(b_melody))
        octave.append(b_octave)
        chords.append(b_chords)

    return {
        'metaData': stream.getMetaAnalysis(),
        'rhythm': rhythm,
        'melody': {
            'notes': melody,
            'octaves': octave,
            'chords': chords
        }
    }


class Stream():
    '''
    Holds a list of notes and has some useful functions.
    '''
    def __init__(self, notes=None, ts_num=4):
        self.stream = dict()
        self.offset_times = []
        self.notes = []

        if notes:
            for n in notes:
                self.append(n)

        self.numberOfNotes = len(self.notes)
        self.ts_num = ts_num

    def append(self, note):
        '''Adds a Note to the end of the stream'''
        if self.numberOfNotes > 0:
            self.getLastNote().setNextNote(note)

        offset = note.getOffset()
        self.stream[offset] = note
        self.offset_times.append(offset)

        self.notes.append(note)
        self.numberOfNotes = len(self.notes)

    def recalculate(self):
        '''Make sure lists in order and notes all have durations'''
        self.offset_times = sorted(list(self.stream.keys()))
        self.notes = []
        for ot in self.offset_times:
            self.notes.append(self.stream[ot])
        self.numberOfNotes = len(self.notes)

    def getStartNote(self):
        '''Returns the start note'''
        if len(self.offset_times) > 0:
            return self.stream[self.offset_times[0]]
        else:
            return None

    def getLastNote(self):
        '''Returns the last note'''
        if len(self.offset_times) == 0:
            return None
        return self.stream[self.offset_times[-1]]

    def getNextBarNumber(self):
        '''Returns the next bar number'''
        if self.getLastNote():
            return self.getLastNote().bar + 1
        else:
            return 0

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

        if len(bar) == 0:
            # rest bar...
            rest = Note(-1, next_bar, 0)
            self.append(rest)
            return

        for offset in bar.keys():
            n = bar[offset]
            #offset = float('{:.02f}'.format(offset))
            if n == 'EOP':
                continue
            note = Note(n, next_bar, offset)
            self.append(note)

    def getIndexAtOffset(self, offset):
        ''' Returns the index of the lowest offset above a given offset time '''
        for i, o in enumerate(self.offset_times):
            if o >= offset:
                return i

        return None

    def removeBarsToEnd(self, bar_num):
        ''' Removes all the bars from bar_num onwards '''
        start_index = self.getIndexAtOffset(bar_num * self.ts_num)
        for offset in self.offset_times[start_index:]:
            del self.stream[offset]

        self.offset_times = self.offset_times[:start_index]
        del self.notes[start_index:]

        self.recalculate()


    def readBarData(self, notes, octaves, rhythm, chords):
        ''' Converts the separate melody, rhythm and chord data into a new bar
        and appends it to the current stream'''

        if len(rhythm) != self.ts_num:
            print('Wrong time signature for rhythm, attempting to fix...')
            print(f'(Expected {self.ts_num}, got {len(rhythm)})')
            self.ts_num = len(rhythm)

        RES = len(notes)
        beat_slot = RES/self.ts_num

        r_mask = [False]*RES
        for i, beat in enumerate(rhythm):
            for j in beat:
                index = round((i + j)*beat_slot)
                r_mask[index] = True

        bar = {}
        for i in range(RES):
            if r_mask[i]:
                offset = i*self.ts_num/RES
                pc = notes[i]
                o = octaves[i]
                bar[offset] = (pc, o)

        self.appendBar(bar)

    def getBar(self, num):
        ''' Returns a given bar, in format {offset: Note} '''
        b_notes = [n for n in self.notes if n.bar==num]
        bar = dict([(n.beat, n) for n in b_notes])
        return bar

    def getBars(self):
        ''' Returns list of seperate bars '''
        bars = []
        for i in range(len(self)):
            bars.append(self.getBar(i))
        return bars

    def getNotePlaying(self, bar, beat):
        '''Returns the note that is currently playing at bar, beat'''
        offset = self.ts_num * bar + beat
        #try:
        #    off = max([o for o in self.offset_times if o < offset ])
        #    return self.stream[off]
        #except ValueError:
        #    return None
        earlier_notes = [n for n in self.notes if n.getOffset() <= offset]
        if len(earlier_notes) > 0:
            return earlier_notes[-1]
        else:
            return None

    def getMetaAnalysis(self):
        '''
        Meta analysis is :
            - span: range between highest and lowest note
            - jump: average interval between notes
            - cDens: proportion of chords
            - cDepth: average depth of chords
            - tCent: average MIDI value
            - rDens: rhythmic density
        '''
        analysis = dict()
        midi_pitches = list([n.midi for n in self.notes if not n.isRest])
        span = max(midi_pitches) - min(midi_pitches)
        tonalCenter = sum(midi_pitches) / len(midi_pitches)
        ints = [abs(i-j) for i,j in zip(midi_pitches[:-1], midi_pitches[1:])]
        avgInts = sum(ints) / len(ints)

        density = len(self.notes) / (self.__len__() * self.notes[0].ts_num)

        chordCount = len([n for n in self.notes if n.isChord()])
        chordDep = sum([len(n.chord) for n in self.notes if n.isChord])

        if self.numberOfNotes > 0:
            cDens = chordCount / self.numberOfNotes
        else:
            cDens = 0

        if chordCount > 0:
            cDepth = chordDep/chordCount
        else:
            cDepth = 0

        analysis = {
            'span': span,
            'jump': avgInts,
            'cDens': cDens,
            'cDepth': cDepth,
            'tCent': tonalCenter,
            'rDens': density
        }
        return analysis

    def __len__(self):
        '''Length is the total number of bars'''
        if self.numberOfNotes == 0:
            return 0
        start_bar = self.getStartNote().bar
        end_bar = self.getLastNote().bar
        return end_bar - start_bar + 1


class Note():
    '''
    A Note object holds information on MIDI value, bar number and beat offset,
    plus the next note in the stream.
    Also holds chord info.
    '''
    def __init__(self, note, bar, beat, chord=None, ts_num=4, ts_den=4, division=6,
                 next_note=None, word=None):
        '''
        - Note can be a single integer (MIDI value) or (Pitch Class, Octave) tuple.
        - Chord is a tuple of offsets from root, where self.midi is root, e.g. (0, 4, 7) for major
        '''

        self.rest = False
        #print(type(note))
        if isinstance(note, int) or isinstance(note, float):
            if note < 0:
                self.rest = True
                self.midi = -1
            else:
                self.midi = note
        else:
            if isinstance(note, int) or isinstance(note, float):
                self.midi = note
            else:
                self.midi = PCOctaveToMIDI(note[0], note[1])
        self.bar = bar
        self.beat = round(division*beat) / division
        if chord:
            self.chord = tuple(chord)
        else:
            self.chord = None
        self.ts_num = ts_num
        self.ts_den = ts_den
        self.next_note = next_note

        self.played = False
        self.drawn = False

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

    def isChord(self):
        return self.chord is not None

    def isRest(self):
        return self.rest

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
        if self.rest:
            return f'Rest  @ {self.bar}:{self.beat} ({ self.getDuration() }, {self.chord})'
        else:
            return f'{self.midi} @ {self.bar}:{self.beat} ({ self.getDuration() }, {self.chord})'

import random

TEST_STREAM = Stream()
for i in range(4):
    for j in [0.0, 1.0, 2.0, 2.5, 3.0, 3.5]:
        TEST_STREAM.append(Note(random.randint(45, 65), i, j))

TEST_STREAM.append(Note(60, 4, 0.0, chord=[0, 4, 7]))




