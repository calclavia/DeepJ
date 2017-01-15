import numpy as np
from keras.layers import Dense, Input, merge, Activation, Flatten
from keras.layers.recurrent import LSTM
from music import NUM_CLASSES

def rnn_model():
    # TODO: Use LSTM
    note_input = Input(shape=(NUM_CLASSES,), name='note_input')
    x = note_input
    x = Dense(512, name='h0')(x)
    return [note_input], x


def note_preprocess(env, x):
    arr = np.zeros((NUM_CLASSES,))
    arr[x] = 1
    return arr
