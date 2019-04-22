import pickle as pkl
from networkEngine import DataReader
from player9C import NNPlayer9C

human_song = './userContexts/intro_suspense.pkl'

params= {
    'um': 2
}

if __name__ == '__main__':
    dr = DataReader(0)
    dr.open_data(human_song)

    player = NNPlayer9C(1)

    indexDict = player.indexDict

    player.rhythm_contexts = list(dr.rhythm_contexts[-4:, :, :])
    player.melody_contexts = dr.melody_contexts[:, -4:, :]

    rhythm = []
    melody = []
    octave = []

    for _ in range(40):
        bar, contexts = player.generate_bar(**params)
        r = [indexDict[b] for b in contexts[0][0]]
        rhythm.append(tuple(r))
        melody.append(tuple(contexts[1][0, 0]))
        octave.append(tuple([4]*48)) 

    data = {
        0: {
            'rhythm': rhythm,
            'melody': {
                'notes': melody,
                'octaves': octave
            }
        }
    }

    with open('./userContexts/continued.pkl', 'wb') as f:
        pkl.dump(data, f)


