import music21 as m21
import pickle as pkl

from core import *

'''
Loads and analyzes the music21 corpus and saves data as a tree:

- musicData (list):

    - id:                       an identifying index for this song

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
            - strengths:        notes beat strength ([0, 1])
            - durations:        quarter note length of note
            - pClass:           pitch class ([0, 11])
            - octave:           ocatave ([-2, 6]?)


'''

def getSongData(i, song):
    songData = dict()

    songData['id'] = i
    songData['corpus'] = 'music21 Core Corpus'

    cSong = cleanScore(song)

    instruments = []
    for j, part in enumerate(cSong.parts):
        print(f'   - part {j}')

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
        partData['metaData'] = metaData
        partData['rhythm']   = rhythmData
        partData['melody']   = melodyDict

        instruments.append(partData)

    songData['instruments'] = instruments
    return songData



musicData = []

corpusPath = m21.corpus.getCorePaths()

print(f'Core corpus has {len(corpusPath)} works')

i = 0

for path in corpusPath:
    if i > 1: break

    ps = str(path)
    if 'demos' in ps or 'theoryExercises' in ps:
        continue

    print(f'Parsing {ps}...')

    song = m21.converter.parse(path)

    if isinstance(song, m21.stream.Opus):
        for s in song.scores:
            songData = getSongData(i, s)
            musicData.append(songData)
            i += 1
    else:
        songData = getSongData(i, song)
        musicData.append(songData)
        i += 1


print('DONE')


















