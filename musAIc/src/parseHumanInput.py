import numpy as np
import matplotlib.pyplot as plt

def dist(x, dt):
    dt2 = x%dt
    return min(dt2, dt-dt2)

def loss(dt, events):
    distance = np.array(list(map(lambda x: dist(x, dt), events)))
    return np.sum(distance**2)/dt

def dtToTemp(dt):
    return 60/(dt*16)

def tempoToDt(tempo):
    return 60/(tempo*16)


if __name__ == '__main__':
    # 80bpm: half quarter quarter whole
    #test_data = list(np.array([0.0, 1/16]) * (4*60)/95)
    test_data = list(np.array([0.0, 8/16, 12/16, 16/16, 32/16]) * (4*60)/70)

    tempos = list(np.arange(60, 141, 1))
    dts = list(map(tempoToDt, tempos))
    losses = list(map(lambda dt: loss(dt, test_data), dts))
    for res in zip(tempos, losses):
        print(res)

    print('Tempo: ', tempos[np.argmin(losses)])

    plt.plot(tempos, losses)
    plt.show()
