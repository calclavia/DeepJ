from music import *
import midi
from rl import A3CAgent
from midi_util import *

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

def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,))
    arr[i] = 1
    return arr

# Define the musical styles
styles = ['data/edm', 'data/southern_rock', 'data/hard_rock', 'data/classical']
NUM_STYLES = len(styles)
