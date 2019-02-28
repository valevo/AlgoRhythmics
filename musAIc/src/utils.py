


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
            index = round((i + j)*beat_slot)
            r_mask[index] = True

    bar = {}
    for i in range(RES):
        if r_mask[i]:
            offset = i*ts_num/RES
            pc = notes[i]
            if pc == None:
                # rest...
                continue
            o = octaves[i]
            bar[offset] = pcOctaveToMIDI(int(pc), o)

    return bar


def pcOctaveToMIDI(pc, octave):
    return 12*(octave+1) + pc


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

        for offset in bar.keys():
            n = bar[offset]
            if n == 'EOP':
                continue
            note = Note(n, next_bar, offset)
            self.append(note)

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
                o = ocataves[i]
                bar[offset] = (pc, o)

        self.appendBar(bar)




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
    def __init__(self, note, bar, beat, ts_num=4, ts_den=4, division=2,
                 next_note=None, word=None):
        ''' Note can be a single integer (MIDI value) or (Pitch Class, Octave)
        tuple '''

        if isinstance(note, int):
            self.midi = note
        else:
            self.midi = pcOctaveToMIDI(note[0], note[1])
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

