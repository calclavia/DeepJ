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
