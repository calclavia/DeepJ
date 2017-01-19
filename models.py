import numpy as np
from keras.layers import Dense, Input, merge, Activation, Flatten
from keras.models import Model
from keras.layers.recurrent import GRU
from music import NUM_CLASSES

def rnn_model(num_actions, timesteps):
    note_input = Input(shape=(timesteps, num_actions), name='note_input')
    x = note_input
    for i in range(1):
        x = GRU(64, return_sequences=True, name='lstm' + str(i))(x)
    x = GRU(64, name='lstm_last')(x)

    # Output layers for policy and value estimations
    policy = Dense(num_actions, activation='softmax', name='policy_output')(x)
    value = Dense(1, activation='linear', name='value_output')(x)
    return Model([note_input], [policy, value])

def note_preprocess(env, x):
    arr = np.zeros((NUM_CLASSES,))
    arr[x] = 1
    return arr
