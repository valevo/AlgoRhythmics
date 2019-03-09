import pickle as pkl

data = []
for i in range(1, 33):
    with open('music_data_{:04d}.pkl'.format(i), 'rb') as f:
        data.extend(pkl.load(f))

print('Data loaded')
print(len(data))

beat_vocab = {}
for song in data:
    if isinstance(song, list):
        #print(song)
        break
    ins = song['instruments']
    if isinstance(ins, str):
        print(ins)
        break
    for i in range(ins):
        rhythm = song[i]['rhythm']
        for bar in rhythm:
            for beat in bar:
                if beat in beat_vocab:
                    beat_vocab[beat] += 1
                else:
                    beat_vocab[beat] = 1

print(beat_vocab)
print(len(beat_vocab))

