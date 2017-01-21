import numpy as np
from keras.layers import Dense, Input, merge, Activation, Flatten
from keras.models import Model
from keras.layers.recurrent import GRU
from util import one_hot
from music import NUM_CLASSES

def note_model(timesteps):
    # TODO Try dropout again?
    note_input = Input(shape=(timesteps, NUM_CLASSES), name='note_input')

    x = note_input

    for i in range(1):
        x = GRU(128, return_sequences=True, name='lstm' + str(i))(x)
        x = Activation('relu')(x)

    x = GRU(128, name='lstm_last')(x)
    x = Activation('relu')(x)

    for i in range(1):
        x = Dense(256, name='dense' + str(i))(x)
        x = Activation('relu')(x)

    # Output layers for policy and value estimations
    # TODO: Is binary a good idea?
    """
    policies = [
        Dense(2, activation='softmax', name='note' + str(i))(x)
        for i in range(num_notes)
    ]
    """
    policies = [Dense(NUM_CLASSES, activation='softmax', name='note_policy')(x)]
    value = Dense(1, activation='linear', name='value_output')(x)
    return Model([note_input], policies + [value])


def note_preprocess(env, x):
    return one_hot(x, NUM_CLASSES)
