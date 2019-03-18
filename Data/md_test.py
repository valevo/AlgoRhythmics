import pickle as pkl

data =[]
for i in range(1, 149):
    with open('music21/music_data_{:04d}.pkl'.format(i), 'rb') as f:
        data.extend(pkl.load(f))

print('Data loaded')
print(len(data))

for song in data:
    ins = song['instruments']
    for i in range(ins):
        for md in song[i]['metaData']:
            if len(md.keys()) != 9:
                print(md)
                print(song['name'])
                break
