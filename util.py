from music import *
import midi
from rl import A3CAgent
from midi_util import *
from preprocess import midi_io, melodies_lib, music_pb2

def make_agent():
    from models import note_model, note_preprocess

    time_steps = 8

    return A3CAgent(
        lambda: note_model(time_steps),
        num_workers=3,
        time_steps=time_steps,
        preprocess=note_preprocess,
        entropy_factor=0.05
    )

def create_beat_data(composition, beats_per_bar):
    """
    Augment the composition with the beat count in a bar it is in.
    """
    beat_patterns = []
    i = 0
    for note in composition:
        beat_pattern = np.zeros((beats_per_bar,))
        beat_pattern[i] = 1
        beat_patterns.append(beat_pattern)
        i = (i + 1) % beats_per_bar
    return beat_patterns

# convert an array of values into a dataset matrix
def create_dataset(data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back)]
        dataX.append(a)
        dataY.append(data[i + look_back])
    return dataX, dataY

def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,))
    arr[i] = 1
    return arr

def process_melody(melody):
    res = []
    for x in melody:
        if x >= 0:
            res.append(x - MIN_NOTE)
        else:
            res.append(abs(x) - 1)
    return res

def load_data(path='data/data_cache.npy'):
    return np.load(path)
