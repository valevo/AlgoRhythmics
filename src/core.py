import numpy as np
import music21 as m21
from copy import deepcopy
from fractions import Fraction

#import numpy.random as rand
import random

NOTES = [str(x) for x in range(12)]
EXP_CORPORA = ['jazzMidi']

def getSongData(song, corpus=None, name=None, verbose=False):
    '''
    Loads and analyzes the music21 corpus and saves data as a tree:

    - musicData (list):
        - 'corpus':                 the corpus name
        - 'name':                   file name
        - 'instruments':            number of instruments
        - [id]:
            - metaData:             of the last 4 bars (first 4 the same)
                - ts:               the (initial) time signature
                - length:           number of measures
                - span:             range of notes
                - jump:             anverage note jump
                - cDens:            percentage of chords used
                - cDepth:           average number of notes in the chord
                - tCent:            average note value
                - rDens:            average number of events ber beat
                - pos:              percentage progress in song
                - expression:       1 if human player, 0 otherwise

            - rhythm:               nested list of all beats and their divisions

            - melody:
                - notes:            tuple of measure notes
                - octaves:          tuple of measure octaves
                - chords:           list of chord tuples in each measaure

    '''

    songData = dict()

    songData['corpus'] = corpus
    songData['name'] = name
    songData['instruments'] = len(song.parts)

    cSong = cleanScore(song, verbose=verbose)

    for j, part in enumerate(cSong.parts):
        partData = dict()

        rhythmData = parseRhythmData(part, verbose=verbose)
        melodyData = parseMelodyData(part, verbose=verbose)
        metaData   = metaAnalysis(part, rhythmData, melodyData)

        partData['metaData'] = metaData
        #if corpus in EXP_CORPORA:
        #    partData['melodyData']['expression'] = 1
        partData['rhythm']   = rhythmData
        partData['melody']   = melodyData

        songData[j] = partData

    return songData


def cleanScore(score, quantise=True, verbose=False):
    '''
    - Transposes to key of C
    - Sets simultaneous notes to a chord object (chordifies parts, any problems?)
    - Pads short bars with rests
    - Fixes bars that are too long (assumes notes in correct offset)
    - Strips unnecessary elements
    - Quantise?
    - asserts one time signature for easier batching
    '''

    # what is wanted in each part
    allowed = [
        #m21.instrument.Instrument,
        m21.stream.Measure,
        m21.meter.TimeSignature,
        m21.clef.Clef,
        m21.key.KeySignature,
        m21.note.Note,
        m21.note.Rest,
        m21.chord.Chord
    ]

    if verbose: print('Cleaning...')

    key = score.analyze('key')
    key_diff = m21.interval.Interval(key.tonic, m21.pitch.Pitch('C'))
    if verbose: print('Found Key ({})...'.format(key))

    new_score = m21.stream.Score()

    if not isinstance(score, m21.stream.Score):
        s = m21.stream.Score()
        p = m21.stream.Part()
        p.append(score)
        s.append(p)
        score = s
        print(score)


    for part in score.parts:
        if verbose: print(part, '----------')

        # check if percussive clef...
        if part.recurse().getElementsByClass(m21.clef.PercussionClef):
            if verbose: print('Percussion, skipping!')
            continue

        new_part = m21.stream.Part()

        # make sure of time signature...
        #ts = part.recurse().timeSignature
        #if not ts:
        #    if verbose: print('[CleanSong] Adding time signature...')
        #    part.flat.insert(0, m21.meter.TimeSignature('4/4'))
        #    part.makeMeasures(inPlace=True)

        # force single time signature....
        t_sigs = part.recurse().getElementsByClass(m21.meter.TimeSignature)
        if len(t_sigs) == 0:
            part.timeSignature = m21.meter.TimeSignature('4/4')
            part.makeMeasures(inPlace=True)
            ts = m21.meter.TimeSignature('4/4')
        elif len(t_sigs) > 1:
            part.remove(t_sigs[1:], recurse=True)
            part.makeMeasures(inPlace=True)
            ts = t_sigs[0]
        else:
            ts = t_sigs[0]

        if quantise:
            level = 48 // 12
            score.quantize(quarterLengthDivisors=(level,), inPlace=True)

        # make sure there are measures...
        if not part.hasMeasures():
            if verbose: print('making measures...')
            part.makeMeasures(inPlace=True)

        # get instrument information
        #ins = part.recurse().getInstrument()
        #if ins:
        #    new_part.append(ins)

        # clean each measure...
        for m in part.recurse().getElementsByClass(m21.stream.Measure):

            new_m = m21.stream.Measure(quarterLength=m.quarterLength)

            # strip to core elements...
            for element in m.recurse().getElementsByClass(allowed):
                if isinstance(element, m21.chord.Chord):
                    if len(element.pitches) == 1:
                        note = m21.note.Note(element.pitches[0])
                        note.duration = element.duration
                        new_m.append(note)
                        continue

                new_m.append(element)

            if len(new_m.notesAndRests) == len(new_m.getElementsByClass(m21.note.Rest)):
                if verbose: print('rest bar found')
                new_m = m21.stream.Measure(quarterLength=m.quarterLength)
                new_m.insert(0.0, m21.note.Rest(m.barDuration))
                new_part.append(new_m)
                continue

            if new_m.duration.quarterLength < new_m.barDuration.quarterLength:
                if verbose: print('padding ', m.number)

                if m.number < 2:
                    # need to pad out the left side with a rest...
                    shift = m.paddingLeft
                    r = m21.note.Rest(quarterLength=shift)
                    new_m.insertAndShift(0, r)
                    for element in new_m.elements:
                        if element not in new_m.notesAndRests:
                            element.offset = 0.0
                else:
                    # pad on the right...
                    r = m21.note.Rest(quarterLength=m.paddingRight)
                    new_m.append(r)


            elif m.duration.quarterLength > m.barDuration.quarterLength:
                if verbose: print('measure {} too long...'.format(m.number))

                pass

            new_part.append(new_m)

        new_part.timeSignature = ts
        #new_part.makeMeasures(inPlace=True)
        #new_part.makeNotation(inPlace=True)

        new_score.insert(part.offset, new_part)

    return new_score.transpose(key_diff)


def parseNoteData(part, ts=None, verbose=False):
    '''
    Returns data in form (beat_stength, duration, pitch_class, octave)
    where pitch_class is -1 for rests, and has (+) operator if part of chord
    '''
    data = []
    last_octave = 4

    def get_octave(pitch, lo):
        if pitch.octave:
            return pitch.octave
        else:
            return lo

    if not ts:
        ts = part.recurse().timeSignature
        if not ts:
            if verbose: print('[ParseNoteData] Setting TS to 4/4')
            ts = m21.meter.TimeSignature('4/4')

    if len(part.getElementsByClass(m21.stream.Measure)) == 0:
        if verbose: print('making measures...')
        part.makeMeasures(inPlace=True)

    for m in part.getElementsByClass("Measure"):
        if m.recurse().timeSignature:
            ts = m.recurse().timeSignature
            if verbose: print('[ParseNoteData] Found ts: ', ts)

        for n in m.flat.notesAndRests:
            bs = ts.getAccentWeight(n.offset, permitMeterModulus=True)
            d = n.duration.quarterLength

            if n.tie:
                if n.tie.type != "start":
                    continue

            if isinstance(n, m21.note.Note):
                last_octave = get_octave(n.pitch, last_octave)
                data.append((bs, d, str(n.pitch.pitchClass), last_octave))

                if verbose: print(n.offset, n, data[-1])

            elif isinstance(n, m21.note.Rest):
                data.append((bs, d, '-1', last_octave))

                if verbose: print(n.offset, n, data[-1])

            elif isinstance(n, m21.chord.Chord):
                ps = n.pitches
                for p in ps[:-1]:
                    last_octave =  get_octave(p, last_octave)
                    data.append((bs, d, str(p.pitchClass) + '+', last_octave))
                    if verbose: print(n.offset, n, data[-1])

                last_octave = get_octave(ps[-1], last_octave)
                data.append((bs, d, str(ps[-1].pitchClass), last_octave))
                if verbose: print(n.offset, n, data[-1])

            else:
                if verbose: print('Something else encountered: ', n)

    return data


def parseMelodyData(part, verbose=False):
    '''
    Returns melody data in continuous form:

    {   'notes': [(n0, n1, ... n48), ...],
        'octaves': [(o0, o1, ... o48), ...],
        'chords':  [ch1, ch1, ... chn]   }

    '''
    notes = []
    octaves = []
    chords = []
    last_octave = 4
    notePool = set([1])
    octavePool = set([last_octave])

    # the number of divisions that have to be filled 24 = 2x3x4
    RES = 48

    def get_note_data(note):
        ''' Returns pitchClass, octave, chord '''
        if isinstance(note, m21.chord.Chord):
            if len(note.normalOrder) == 1:
                return get_note_data(note.pitches[0])
            root = note.root()
            chordOrder = note.normalOrder
            chord = [(pc - chordOrder[0]) for pc in chordOrder]
            return root.pitchClass+1+12, root.octave, chord

        elif isinstance(note, m21.note.Rest):
            return None, None, None

        elif isinstance(note, m21.pitch.Pitch):
            return note.pitchClass+1, note.octave, None

        else:
            return note.pitch.pitchClass+1, note.pitch.octave, None


    ts = part.recurse().timeSignature
    if not ts:
        ts = m21.meter.TimeSignature('4/4')
        if verbose: print('[ParseMelodyData] Setting TS to 4/4')

    if not part.hasMeasures():
        if verbose: print('making measures...')
        part.makeMeasures(inPlace=True)

    #chordPercent = metaData['avgChord']

    for m in part.getElementsByClass("Measure"):
        m_notes = [None]*RES
        m_octaves = [None]*RES
        m_chords = []

        duration = m.duration.quarterLength

        for n in m.flat.notesAndRests:
            # compute notes index...
            idx = round(RES*(n.offset/duration))

            p, o, c = get_note_data(n)

            if p: notePool.add(p)
            if o: octavePool.add(o)

            m_notes[idx] = p
            m_octaves[idx] = o
            if c: m_chords.append(tuple(c))

        ## fill in the gaps...
        for i in range(RES):
            if m_notes[i]:
                continue
            else:
                m_notes[i] = random.sample(notePool, 1)[0]
                #if np.random.rand() < chordPercent:

                m_octaves[i] = random.sample(octavePool, 1)[0]

        notes.append(tuple(m_notes))
        octaves.append(tuple(m_octaves))
        chords.append(m_chords)

    return {'notes': notes, 'octaves': octaves, 'chords': chords}


def parseRhythmData(part, force_ts=None, verbose=False):
    '''
    Splits the score into single beat length 'words' that contain offset and tie information from that beat.
    - Organised as a list of measures that are the list of beats
    - Gets the length of the beat from the Time Signature
    - Robust against odd time signatures and tied notes
    '''

    data = []

    if force_ts:
        ts = force_ts
    else:
        ts = part.recurse().timeSignature
        if not ts:
            ts = m21.meter.TimeSignature('4/4')


    beat_length = Fraction(ts.beatDuration.quarterLength)

    if part.hasMeasures():
        if verbose: print('making measures...')
        part.makeMeasures(inPlace=True)

    for m in part.recurse().getElementsByClass("Measure"):
        measure = []
        if m.recurse().timeSignature and not force_ts:
            ts = m.recurse().timeSignature
            beat_length = Fraction(ts.beatDuration.quarterLength)
            if verbose: print('[ParseRhythmData] New ts: ', ts, beat_length)

        m_length = ts.numerator    # number of words in a measure
        if verbose: print('[parseRhythmData] Measure duration: ', m_length)

        for i in range(m_length):
            offset = i * beat_length
            beat = m.flat.getElementsByOffset(offset, offset+beat_length, includeEndBoundary=False)
            word = set()
            for x in beat.recurse().notesAndRests:
                if x.tie:
                    if x.tie.type != 'start':
                        continue

                onsetTime = round((float(x.offset - offset) / beat_length), 4)
                word.add(onsetTime)

            measure.append(tuple(sorted(word)))
            if verbose: print(tuple(sorted(word)))

        while len(measure) < ts.numerator:
            measure.append(())

        data.append(tuple(measure))

    return data


def metaAnalysis(part, rhythm, melody):
    '''
    Args:
        - stream: to be analysed
    Returns:
        - analysis: dictionary of results, where
            - 'ts':                 time signature (as string)
            - 'length':             total number of measures
            - 'span' :              pitch span
            - 'jump' :              average interval size
            - 'cDens' :             proportion of chords
            - 'cDepth':             average number of notes in chord
            - 'tCent' :             mean MIDI value
            - 'rDens':              average number of events per beat
            - 'expression':         0 if perfect on beat, 1 if played expressively
                                    (corpus dependent)
    '''

    def get_pitch(note):
        if isinstance(note, m21.chord.Chord):
            return note.root().midi
        else:
            return note.pitch.midi

    result = list()

    #mid = m21.analysis.discrete.MelodicIntervalDiversity()

    md = []
    measures = part.recurse().getElementsByClass(m21.stream.Measure)

    for i in range(3, len(measures)):
        analysis = {'ts': '4/4',
                    'span': 1,
                    'jump': 1,
                    'cDens': 0,
                    'cDepth': 1,
                    'tCent': 60,
                    'rDens': 1,
                    'expression': 0}

        section = part.measures(i-3, i, indicesNotNumbers=True)

        notes = section.flat.notes

        if len(notes) == 0:
            # have a score in Chord notation? skip it...
            continue

        timeSig = section.recurse().timeSignature
        if timeSig:
            ts = timeSig.ratioString
        else:
            ts = '4/4'

        length = len(section.getElementsByClass(m21.stream.Measure))

        midiPitches = list(map(get_pitch, notes))

        if len(midiPitches) == 0:
            md.append(analysis)
            continue

        span = max(midiPitches) - min(midiPitches)
        tonalCenter = sum(midiPitches)/len(midiPitches)

        ints = [abs(p1-p2) for p1, p2 in zip(midiPitches[:-1], midiPitches[1:])]
        avgInt = sum(ints)/len(ints)

        chords = section.flat.getElementsByClass(m21.chord.Chord)
        avgChord = len(chords)/len(notes)
        #avgChord = len(melody['chords']) / len(notes)

        if len(chords) == 0:
            chordDepth = 0
        else:
            chordDepth = sum([len(c) for c in chords])/len(chords)

        beatCount = 0
        events = 0
        for m2 in rhythm[i-3:i]:
            for b in m2:
                beatCount += 1
                events += len(b)

        rhythmicDensity = events/beatCount

        pos = i/len(measures)

        analysis = {'ts': ts,
                    'span': span,
                    'jump': avgInt,
                    'cDens': avgChord,
                    'cDepth': chordDepth,
                    'tCent': tonalCenter,
                    'rDens': rhythmicDensity,
                    'pos': pos,
                    'expression': 0}

        #print(analysis)

        md.append(analysis)
        if i == 3:
            # append three more times...
            md *= 4

    return md



