import numpy as np
from keras.layers import Dense, Input, merge, Activation, Flatten
from keras.layers.recurrent import LSTM
from music import NUM_CLASSES


def rnn_model(timesteps):
    note_input = Input(shape=(timesteps, NUM_CLASSES), name='note_input')
    x = note_input
    for i in range(2):
        x = LSTM(64, return_sequences=True, name='lstm' + str(i))(x)
    x = LSTM(64, name='lstm_last')(x)
    return [note_input], x


def note_preprocess(env, x):
    arr = np.zeros((NUM_CLASSES,))
    arr[x] = 1
    return arr
