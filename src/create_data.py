import music21 as m21
import pickle as pkl

import os
import logging

from multiprocessing import Pool, Queue, Process, current_process, cpu_count

from core import *

MAX_WORKERS = cpu_count()

def saveProgress(data, name, path='../Data/'):
    logging.info(f'--- Saving progress at {path}{name}')

    with open(f'{path}{name}', 'wb') as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)



class ParseData(object):
    def __init__(self, corpus='music21'):
        self.jobQ = Queue()
        self.returnQ = Queue()

        self.corpusFunctions = {
            'music21': (self.m21Worker, self.m21Jobs)
        }

        logging.info(f'Starting {MAX_WORKERS} workers...')
        self.pool = Pool(processes=MAX_WORKERS,
                         initializer=self.corpusFunctions[corpus][0])
        self.collector = Process(target=self.collectResults)

        self.temp_buffer = []

        # start collector process...
        self.collector.start()

        self.corpusFunctions[corpus][1]()

        self.pool.close()

        # wait for workers to finish...
        self.pool.join()

        # wait for collector to finish...
        self.collector.join()

    def collectResults(self):
        logging.info('Collector started')
        i = 0
        musicData = []
        started = False
        while True:
            try:
                if started:
                    data = self.returnQ.get(block=True, timeout=60)
                else:
                    data = self.returnQ.get(block=True, timeout=None)
                    started = True

            except:
                # assume no more results, finish up
                name = 'music21/music_data_{:04d}.pkl'.format(i//100+1)
                saveProgress(musicData, name)
                return

            musicData.append(data)
            i += 1
            if i%100 == 0:
                name = 'music21/music_data_{:04d}.pkl'.format(i//100)
                saveProgress(musicData, name)
                musicData = []

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
        logging.info(f'Worker started [PID: {current_process().pid}]')
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




if __name__ == '__main__':

    logging.basicConfig(filename='./createData.log', level=logging.INFO,
                       format='%(asctime)s %(message)s')

    logging.info('START')

    data_parser = ParseData(corpus='music21')

    logging.info('DONE')


