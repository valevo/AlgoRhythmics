import numpy as np
import music21 as m21
from copy import deepcopy
from fractions import Fraction

NOTES = [str(x) for x in range(12)]


def getSongData(song):
    '''
    Loads and analyzes the music21 corpus and saves data as a tree:

    - musicData (list):
        - corpus:                   the corpus name

        - instruments:              list of instruments
            - id:                   index of instrument

            - metaData:
                - ts:               the (initial) time signature
                - length:           number of measures
                - span:             range of notes
                - avgInt:           anverage note jump
                - avgChord:         percentage of chords used
                - tonalCenter:      average note value

            - rhythm:               nested list of all beats and their divisions

            - melody:
                - notes:            tuple of measure notes
                - octaves:          tuple of measure octaves

    '''

    songData = dict()

    songData['corpus'] = 'music21 Core Corpus'

    cSong = cleanScore(song)

    instruments = []
    for j, part in enumerate(cSong.parts):
        partData = dict()

        rhythmData = parseRhythmData(part)
        melodyData = parseNoteData(part)
        metaData   = metaAnalysis(part)

        strengths = []
        durations = []
        pClass    = []
        octaves   = []

        for note in melodyData:
            strengths.append(note[0])
            durations.append(note[1])
            pClass.append(note[2])
            octaves.append(note[3])

        melodyDict = dict()
        melodyDict['strengths'] = strengths
        melodyDict['durations'] = durations
        melodyDict['pClass']    = pClass
        melodyDict['octaves']   = octaves

        partData['id']       = j
        partData['metaData'] = metaData[0]
        partData['rhythm']   = rhythmData
        partData['melody']   = melodyDict

        instruments.append(partData)

    songData['instruments'] = instruments
    return songData


def cleanScore(score, verbose=False):
    '''
    - Transposes to key of C
    - Sets simultaneous notes to a chord object (chordifies parts, any problems?)
    - Pads short bars with rests
    - Fixes bars that are too long (assumes notes in correct offset)
    - Strips unnecessary elements
    - etc
    '''

    # what is wanted in each part
    allowed = [
        m21.instrument.Instrument,
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
#         score.makeNotation(inPlace=True)

    for part in score.parts:
        if verbose: print(part, '----------')

        # check if percussive clef...
        if part.recurse().getElementsByClass(m21.clef.PercussionClef):
            if verbose: print('Percussion, skipping!')
            continue

        new_part = m21.stream.Part()

        # make sure of time signature...
        if not part.flat.timeSignature:
            part.flat.insert(0, m21.meter.TimeSignature('4/4'))
            part.makeMeasures(inPlace=True)

        # make sure there are measures...
        if part.hasMeasures():
            if verbose: print('making measures...')
            part.makeMeasures(inPlace=True)

        # get instrument information
        ins = part.recurse().getInstrument()
        if ins:
            new_part.append(ins)

        # clean each measure...
        for m in part.getElementsByClass(m21.stream.Measure):

            new_m = m21.stream.Measure(quarterLength=m.quarterLength)

            # strip to core elements...
            for element in m.chordify().flat.getElementsByClass(allowed):
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

        new_part.makeMeasures(inPlace=True)
        new_part.makeNotation(inPlace=True)

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
        ts = part.timeSignature
        if not ts:
            ts = m21.meter.TimeSignature('4/4')

    if len(part.getElementsByClass(m21.stream.Measure)) == 0:
        if verbose: print('making measures...')
        part.makeMeasures(inPlace=True)

    for m in part.getElementsByClass("Measure"):
        if m.timeSignature:
            ts = m.timeSignature

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


def parseMelodyData(part, metaData, verbose=False):
    '''
    Returns melody data in continuous form:
    {
        'notes': [(n0, n1, ... n31), ...],
        'octaves': [(o0, o1, ... o31), ...]

    }
    where pitch_class is -1 for rests, and has (+) operator if part of chord
    '''
    notes = []
    octaves = []
    last_octave = 4

    # the number of divisions that have to be filled 24 = 2x3x4
    RES = 24

    def get_octave(pitch, lo):
        if pitch.octave:
            return pitch.octave
        else:
            return lo

    def get_note_data(note):
        if isinstance(note, m21.chord.Chord):
            tonic = note.pitches[0]
            return str(tonic.pitchClass) + '+', tonic.octave
        elif isinstance(note, m21.note.Rest):
            return None, None
        else:
            return str(note.pitch.pitchClass), note.pitch.octave


    ts = part.timeSignature
    if not ts:
        ts = m21.meter.TimeSignature('4/4')

    if not part.hasMeasures():
        if verbose: print('making measures...')
        part.makeMeasures(inPlace=True)

    chordPercent = metaData['avgChord']

    for m in part.getElementsByClass("Measure"):
        m_notes = [None]*RES
        m_octaves = [None]*RES

        duration = m.duration.quarterLength

        lo_o = 4
        hi_o = 4

        for n in m.flat.notesAndRests:
            # compute notes index...
            idx = round(RES*(n.offset/duration))

            p, o = get_note_data(n)

            m_notes[idx] = p
            m_octaves[idx] = o

            if o:
                lo_o = min(o, lo_o)
                hi_o = max(o, hi_o)


        # fill in the gaps...
        # complete random, of slightly more thoughtful?

        for i in range(RES):
            if m_notes[i]:
                continue
            else:
                m_notes[i] = str(np.random.randint(12))
                if np.random.rand() < chordPercent:
                    m_notes[i] = m_notes[i] + '+'

                m_octaves[i] = np.random.randint(lo_o, hi_o+1)

        notes.append(tuple(m_notes))
        octaves.append(tuple(m_octaves))

    return {'notes': notes, 'octaves': octaves}


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
        ts = m21.meter.TimeSignature('4/4')

    beat_length = Fraction(ts.beatDuration.quarterLength)

    if len(part.getElementsByClass(m21.stream.Measure)) == 0:
        if verbose: print('making measures...')
        part.makeMeasures(inPlace=True)

    for m in part.getElementsByClass("Measure"):
        measure = []
        if m.timeSignature and not force_ts:
            ts = m.timeSignature
            beat_length = Fraction(ts.beatDuration.quarterLength)
            if verbose: print('New ts: ', ts, beat_length)

        m_length = ts.numerator    # number of words in a measure
        if verbose: print('Measure duration: ', m_length)

        for i in range(m_length):
            offset = i * beat_length
            w = m.flat.getElementsByOffset(offset, offset+beat_length, includeEndBoundary=False)
            word = []
            for x in w.flat.notesAndRests:
                if x.tie:
                    if x.tie.type != 'start':
                        continue

                word.append(x.offset - offset)

            measure.append(tuple(word))
            if verbose: print(word)

        while len(measure) < ts.numerator:
            measure.append(())

        data.append(tuple(measure))

    return data


def metaAnalysis(stream):
    '''
    Args:
        - stream: to be analysed
    Returns:
        - analysis: dictionary of results, where
            - 'ts':             time signature (as string)
            - 'length':         total number of measures
            - 'span' :          pitch span
            - 'avgInt' :        average interval size
            - 'avgChord' :      proportion of chords
            - 'tonalCenter' :   mean MIDI value
            - 'expression':     0 if perfect on beat, 1 if played expressively
                                (corpus dependent)
    '''

    def get_pitch(note):
        if isinstance(note, m21.chord.Chord):
            return note.root().midi
        else:
            return note.pitch.midi

    result = list()

    if isinstance(stream, m21.stream.Score):
        parts = stream.parts
    else:
        parts = [stream]

    mid = m21.analysis.discrete.MelodicIntervalDiversity()

    for part in parts:
        analysis = dict()

        notes = part.flat.notes

        if len(notes) == 0:
            # have a score in Chord notation? skip it...
            continue

        timeSig = part.flat.timeSignature
        if timeSig:
            ts = timeSig.ratioString
        else:
            ts = '4/4'

        length = len(part.getElementsByClass(m21.stream.Measure))

        midiPitches = list(map(get_pitch, notes))

        span = max(midiPitches) - min(midiPitches)
        tonalCenter = sum(midiPitches)/len(midiPitches)

        ints = [abs(i-j) for i, j in zip(midiPitches[:-1], midiPitches[1:])]
        avgInt = sum(ints)/len(ints)

        avgChord = len(part.flat.getElementsByClass(m21.chord.Chord))/len(notes)

        analysis = {'ts': ts,
                    'span': span,
                    'avgInt': avgInt,
                    'avgChord': avgChord,
                    'tonalCenter': tonalCenter,
                    'expression': 0}

        result.append(analysis)

    return result



