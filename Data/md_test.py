import pickle as pkl
from fractions import Fraction
import numpy as np

data =[]
for i in range(1, 149):
    with open('music21/music_data_{:04d}.pkl'.format(i), 'rb') as f:
        data.extend(pkl.load(f))

print('Data loaded')
print(len(data))

def prepare_metaData(metaData, repeat=0):
    values = []
    meta_keys = metaData.keys()

    for k in meta_keys:
        if k == "ts":
            frac = Fraction(metaData[k], _normalize=False)
            values.extend([frac.numerator, frac.denominator])
        else:
            assert isinstance(metaData[k], (float, int))
            values.append(metaData[k])

    if len(values) != 10:
        raise ValueError("DataGenerator.prepare_metaData: Expected metaData of length 10," +
                         " recieved length {}, \nMetaData: {}".format(len(values), metaData))

    if not repeat:
        return np.asarray(values, dtype="float")
    else:
        return np.repeat(np.asarray([values], dtype="float"), repeat, axis=0)

for song in data:
    ins = song['instruments']
    for i in range(ins):
        if len(song[i]['metaData']) == 0:
            print(song['name'])
            break
        for md in song[i]['metaData']:
            if prepare_metaData(md).shape != (10,):
                print(md, prepare_metaData(md))
                print(song['name'])
                break
