# Defines the models used in the experiments

import numpy as np
from keras.layers import Dense, Input, merge, Activation, Dropout, Flatten
from keras.models import Model
from keras.layers.convolutional import Convolution1D
from keras.layers.recurrent import GRU
from util import one_hot
from music import NUM_CLASSES, NOTES_PER_BAR
from keras.models import load_model

def pre_model(time_steps, dropout=True):
    # Multi-hot vector of each note
    # TODO: Just change this to state for simplicity.
    note_input = Input(shape=(time_steps, NUM_CLASSES), name='note_input')
    beat_input = Input(shape=(time_steps, NOTES_PER_BAR), name='beat_input')
    num_units = 256

    x = note_input
    y = GRU(64, return_sequences=True, name='beat_sparse')(beat_input)

    if dropout:
        x = Dropout(0.2)(x)

    x = merge([x, y], mode='concat')

    for i in range(2):
        x = GRU(num_units, return_sequences=True, name='lstm' + str(i))(x)
        x = Activation('relu')(x)
        if dropout:
            x = Dropout(0.5)(x)

    x = GRU(num_units, name='lstm_last')(x)
    x = Activation('relu')(x)
    if dropout:
        x = Dropout(0.5)(x)

    for i in range(1):
        x = Dense(num_units * 2, name='dense' + str(i))(x)
        x = Activation('relu')(x)
        if dropout:
            x = Dropout(0.5)(x)

    return [note_input, beat_input], x

def note_model(time_steps):
    inputs, x = pre_model(time_steps, False)

    # Multi-label
    policy = Dense(NUM_CLASSES, name='policy', activation='softmax')(x)
    value = Dense(1, name='value', activation='linear')(x)

    model = Model(inputs, [policy, value])
    model.load_weights('data/supervised.h5', by_name=True)
    # Create value output
    return model


def supervised_model(time_steps):
    inputs, x = pre_model(time_steps, False)

    # Multi-label
    x = Dense(NUM_CLASSES, name='policy', activation='softmax')(x)

    model = Model(inputs, x)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def note_preprocess(env, x):
    note, beat = x
    return (one_hot(note, NUM_CLASSES), one_hot(beat, NOTES_PER_BAR))
