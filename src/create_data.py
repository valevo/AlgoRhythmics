import music21 as m21
import pickle as pkl

import os
import logging

from multiprocessing import Pool, Queue, Process, current_process, cpu_count

from core import *

MAX_WORKERS = cpu_count()



class ParseData(object):
    def __init__(self, corpus='music21', outPath='../Data/', **kwargs):
        self.corpus = corpus
        self.outPath = outPath
        self.kwargs = kwargs

        # make sure output directory exists
        if not os.path.isdir(f'{self.outPath}{self.corpus}'):
            logging.info(f'Output path created: {self.outPath}{self.corpus}/')
            os.makedirs(f'{self.outPath}{self.corpus}', exist_ok=True)

        self.corpusFunctions = {
            'music21': (self.m21Worker, self.m21Jobs),
            'jazzMidi'   : (self.midiWorker, self.midiJobs)
        }

        self.jobQ = Queue()
        self.returnQ = Queue()
        self.workersRunning = True

        logging.info(f'Starting {MAX_WORKERS} workers...')
        self.pool = Pool(processes=MAX_WORKERS,
                         initializer=self.corpusFunctions[corpus][0])
        self.collector = Process(target=self.collectResults)

        self.temp_buffer = []

        # start collector process...
        self.collector.start()

        self.corpusFunctions[corpus][1]()

        # no more workers will be added...
        self.pool.close()

        # wait for workers to finish...
        self.pool.join()
        self.workersRunning = False

        # wait for collector to finish...
        self.collector.join()


    def saveProgress(self, data, name):
        logging.info(f'--- Saving progress at {self.outPath}{name}')

        with open(f'{self.outPath}{name}', 'wb') as f:
            pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)


    def collectResults(self):
        logging.info('Collector started')
        i = 0
        musicData = []
        started = False

        SAVE_FREQ = 10

        while True:
            if self.workersRunning:
                # workers running, check for done data
                try:
                    data = self.returnQ.get(block=True, timeout=10)
                except:
                    continue

                musicData.append(data)
                i += 1
                if i%SAVE_FREQ == 0:
                    name = '{}/music_data_{:04d}.pkl'.format(self.corpus,
                                                             i//SAVE_FREQ)
                    self.saveProgress(musicData, name)
                    musicData = []

            else:
                if self.returnQ.empty():
                    # workers done, no more data, finish up
                    logging.info('Finishing up...')

                    name = '{}/music_data_{:04d}.pkl'.format(self.corpus,
                                                             i//SAVE_FREQ+1)
                    self.saveProgress(musicData, name)
                    logging.info('Collector finished.')
                    return

                else:
                    # workers done, but queue not empty...
                    data = self.returnQ.get(block=False)
                    if not data:
                        continue

                    musicData.append(data)
                    i += 1
                    if i%SAVE_FREQ == 0:
                        name = '{}/music_data_{:04d}.pkl'.format(self.corpus,
                                                                 i//SAVE_FREQ)
                        self.saveProgress(musicData, name)
                        musicData = []


            #try:
            #    if started:
            #        data = self.returnQ.get(block=True, timeout=60)
            #    else:
            #        data = self.returnQ.get(block=True, timeout=None)
            #        started = True

            #except:
            #    # assume no more results, finish up
            #    name = '{}/music_data_{:04d}.pkl'.format(self.corpus,
            #                                             i//SAVE_FREQ+1)
            #    self.saveProgress(musicData, name)
            #    logging.info('Collector finishing...')
            #    return

            #musicData.append(data)
            #i += 1
            #if i%SAVE_FREQ == 0:
            #    name = '{}/music_data_{:04d}.pkl'.format(self.corpus,
            #                                             i//SAVE_FREQ)
            #    self.saveProgress(musicData, name)
            #    musicData = []


    def addJobToQueue(self, msg):
        if self.jobQ.full():
            self.temp_buffer.append(msg)
        else:
            self.jobQ.put(msg)
            if len(self.temp_buffer) > 0:
                addJobToQueue(self.temp_buffer.pop())

    def m21Jobs(self):
        ''' Add jobs from the music21 corpus '''
        corpusPath = m21.corpus.getCorePaths()
        logging.info(f'Core corpus has {len(corpusPath)} works')

        for path in corpusPath:
            self.jobQ.put(path, block=True, timeout=None)

        logging.info('All jobs queued')

    def m21Worker(self):
        ''' Parses data from the music21 corpus '''
        logging.info(f'm21 Worker started [PID: {current_process().pid}]')
        started = False
        while True:
            if not started:
                path = self.jobQ.get(block=True, timeout=None)
                started = True
            else:
                if self.jobQ.empty():
                    logging.info('Worker finished')
                    return

                path = self.jobQ.get(block=True, timeout=None)

            ps = str(path)
            if 'demos' in ps or 'theoryExercises' in ps:
                continue

            try:
                logging.info(f'Parsing {ps}...')
                song = m21.converter.parse(path)

                if isinstance(song, m21.stream.Opus):
                    logging.info(f'(OPUS of size {len(song.scores)})')
                    for s in song.scores:
                        songData = getSongData(s, corpus='music21')
                        self.returnQ.put(songData)

                else:
                    songData = getSongData(song, corpus='music21')
                    self.returnQ.put(songData)

            except:
                logging.exception('')


    def midiJobs(self):
        ''' Add jobs from MIDI folder, specified by folder '''
        if self.kwargs['folder']:
            folder = self.kwargs['folder']
            logging.info(f'MIDI folder to parse: {folder}')
        else:
            logging.warning('No folder found for MIDI. Aborting...')
            return

        for path, subdirs, files in os.walk(folder):
            for name in files:
                self.jobQ.put(os.path.join(path, name), block=True,
                              timeout=None)

        return

    def midiWorker(self):
        ''' Parse MIDI paths '''
        logging.info(f'MIDI Worker started [PID: {current_process().pid}]')
        started = False
        while True:
            if not started:
                path = self.jobQ.get(block=True, timeout=None)
                started = True
            else:
                if self.jobQ.empty():
                    logging.info('Worker finished')
                    return

                path = self.jobQ.get(block=True, timeout=None)

            ps = str(path)
            try:
                logging.info(f'Parsing {ps}...')
                song = m21.converter.parse(path)

                # quantize song
                # IMPORTANT: Settle on nice divisor!
                # is Melody res = 24, then (6,), if res = 32, then (8,)
                song.quantize(quarterLengthDivisors=(6,), inPlace=True)

                # remove duplicate voices?
                song2 = m21.stream.Score()
                for p in song.parts:
                    song2.insert(0.0, p.chordify())

                songData = getSongData(song2, corpus=self.corpus)
                self.returnQ.put(songData)

            except:
                logging.exception('')

        return




if __name__ == '__main__':

    logging.basicConfig(filename='./createData.log', level=logging.INFO,
                       format='%(asctime)s %(message)s')

    logging.info('START')

    data_parser = ParseData(corpus='jazzMidi', folder='../SourceData/JazzMidi/')
    #data_parser = ParseData(corpus='music21')


    logging.info('DONE')


