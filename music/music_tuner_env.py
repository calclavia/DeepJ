import numpy as np
import random
from .music_theory_env import MusicTheoryEnv
from .util import NUM_CLASSES, NOTES_PER_BAR

from rl import Memory
# Keras supervised model
from keras.models import load_model
from keras import backend as K
import tensorflow as tf

# TODO: Don't hardcode this
g_rnn = tf.Graph()
with g_rnn.as_default():
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
        # Separate the graph!
        self.g_rnn = g_rnn
        self.supervised_model = supervised_model

        self.theory_scalar = theory_scalar
        self.time_steps = 8
        self.memory = Memory(self.time_steps)

    def _reset(self):
        state = super()._reset()
        # TODO: Don't hardcode!
        self.preprocess = note_preprocess
        self.memory.reset(self.preprocess(self, state))
        return state

    def _step(self, action):
        # Ask the Melody RNN to make a prediction
        with self.g_rnn.as_default():
            s = [np.array([s_i]) for s_i in self.memory.to_states()]
            predictions = self.supervised_model.predict(s)[0]
            norm_constant = np.log(np.sum(np.exp(predictions)))
            prob = predictions[action] - norm_constant #np.log(np.clip(predictions[action], 1e-20, 1))

        # Compute music theory rewards
        state, reward, done, info = super()._step(action)

        # Total reward = log(P(a | s)) + r_TM * c
        reward = prob #+ reward * self.theory_scalar

        self.memory.remember(self.preprocess(self, state))
        return state, reward, done, info
