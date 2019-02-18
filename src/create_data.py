import music21 as m21
import pickle as pkl

import logging

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
        #print(f'   - part {j}')

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

def saveProgress(data, name, path='./data/'):
    logging.info(f'--- Saving progress at {path}{name}')

    with open(f'{path}{name}', 'wb') as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)


logging.basicConfig(filename='./createData.log', level=logging.INFO,
                   format='%(asctime)s %(message)s')

musicData = []

corpusPath = m21.corpus.getCorePaths()

logging.info(f'Core corpus has {len(corpusPath)} works')

i = 0

for path in corpusPath:
    ps = str(path)
    if 'demos' in ps or 'theoryExercises' in ps:
        continue

    logging.info(f'Parsing {ps}...[{i}]')

    try:
        song = m21.converter.parse(path)

        if isinstance(song, m21.stream.Opus):
            logging.info(f'(OPUS of size {len(song.scores)})')
            for s in song.scores:
                songData = getSongData(i, s)
                musicData.append(songData)
                i += 1

                if i%100 == 0:
                    name = 'music_data_{:04d}.pkl'.format(i//100)
                    saveProgress(musicData, name)
                    musicData = []
        else:
            songData = getSongData(i, song)
            musicData.append(songData)
            i += 1

            if i%100 == 0:
                name = 'music_data_{:04d}.pkl'.format(i//100)
                saveProgress(musicData, name)
                musicData = []

    except:
        logging.exception('')

name = 'music_data_{:04d}.pkl'.format(i//100 + 1)
saveProgress(musicData, name)

logging.info('DONE')



















