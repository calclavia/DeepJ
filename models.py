import numpy as np
from keras.layers import Dense, Input, merge, Activation, Flatten
from keras.models import Model
from keras.layers.recurrent import GRU
from util import one_hot
from music import NUM_CLASSES, NOTES_PER_BAR

def note_model(timesteps):
    # TODO Try dropout again?
    note_input = Input(shape=(timesteps, NUM_CLASSES), name='note_input')
    beat_input = Input(shape=(timesteps, NOTES_PER_BAR), name='beat_input')

    x = merge([note_input, beat_input], mode='concat')

    for i in range(2):
        x = GRU(128, return_sequences=True, name='lstm' + str(i))(x)
        x = Activation('relu')(x)

    x = GRU(128, name='lstm_last')(x)
    x = Activation('relu')(x)

    for i in range(1):
        x = Dense(256, name='dense' + str(i))(x)
        x = Activation('relu')(x)

    policy = Dense(NUM_CLASSES, activation='softmax', name='note_policy')(x)
    value = Dense(1, activation='linear', name='value_output')(x)
    return Model([note_input, beat_input], [policy, value])

def note_preprocess(env, x):
    note, beat = x
    return (one_hot(note, NUM_CLASSES), one_hot(beat, NOTES_PER_BAR))
