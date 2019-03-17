import pickle as pkl

data =[]
for i in range(1, 150):
    with open('music21/music_data_{:04d}.pkl'.format(i), 'rb') as f:
        data.extend(pkl.load(f))

print('Data loaded')
print(len(data))

beat_vocab ={}

for song in data:
    ins = song['instruments']
    for i in range(ins):
        rhythm = song[i]['rhythm']
        for bar in rhythm:
            for beat in bar:
                if beat in beat_vocab:
                    beat_vocab[beat] += 1
                else:
                    beat_vocab[beat] = 1

for k, v in beat_vocab.items():
    print(k, v)

print(len(beat_vocab))
