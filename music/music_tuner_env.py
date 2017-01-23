import numpy as np
import random
from .music_theory_env import MusicTheoryEnv
from .util import NUM_CLASSES, NOTES_PER_BAR

from rl import Memory
# Keras supervised model
from keras.models import load_model
from keras import backend as K
import tensorflow as tf

# Separate the graph!
g_rnn = tf.Graph()
with g_rnn.as_default():
    # TODO: Don't hardcode this
    supervised_model = load_model('data/supervised.h5')

# TODO: Don't copy this... Hacks.
def note_preprocess(env, x):
    note, beat = x
    return (one_hot(note, NUM_CLASSES), one_hot(beat, NOTES_PER_BAR))

def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,))
    arr[i] = 1
    return arr


class MusicTunerEnv(MusicTheoryEnv):
    """
    Music environment that combines tuning rewards and theory rewards
    https://arxiv.org/pdf/1611.02796v4.pdf
    """
    def __init__(self, theory_scalar=1):
        super().__init__()
        self.theory_scalar = theory_scalar

    def _reset(self):
        state = super()._reset()
        # TODO: Don't hardcode!
        self.preprocess = note_preprocess
        self.time_steps = 8
        self.memory = Memory(self.time_steps)
        self.memory.reset(self.preprocess(self, state))
        return state

    def _step(self, action):
        # TODO: Avoid globals!
        global supervised_model

        # Ask the Melody RNN to make a prediction
        with g_rnn.as_default():
            s = [np.array([s_i]) for s_i in self.memory.to_states()]
            predictions = supervised_model.predict(s)[0]
            prob = predictions[action]

        # Compute music theory rewards
        state, reward, done, info = super()._step(action)

        # Total reward = log(P(a | s)) + r_TM * c
        reward = np.log(prob) + reward * self.theory_scalar

        self.memory.remember(self.preprocess(self, state))
        return state, reward, done, info
