import sys

# Import modules
sys.path.append('../gym-music')

from music import *
import midi
from rl import A3CAgent
from models import *

def make_agent():
    time_steps = 5
    return A3CAgent(
        NUM_CLASSES,
        lambda: rnn_model(time_steps),
        time_steps=time_steps,
        preprocess=note_preprocess,
        entropy_factor=1e-1
    )

def create_beat_data(composition, beats_per_bar=BEATS_PER_BAR):
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
