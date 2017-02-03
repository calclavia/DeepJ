# Defines the models used in the experiments

import numpy as np
from keras.layers import Dense, Input, merge, Activation, Dropout, Flatten
from keras.models import Model
from keras.layers.convolutional import Convolution1D
from keras.layers.recurrent import GRU
from util import one_hot
from music import NUM_CLASSES, NOTES_PER_BAR, NUM_KEYS
from keras.models import load_model


def gru_stack(time_steps, dropout=True, num_units=256, layers=3):
    # Multi-hot vector of each note
    note_input = Input(shape=(time_steps, NUM_CLASSES), name='note_input')
    # One hot vector for current beat
    beat_input = Input(shape=(time_steps, NOTES_PER_BAR), name='beat_input')
    # key_input = Input(shape=(time_steps, NUM_KEYS), name='key_input')

    x1 = note_input
    x2 = beat_input
    # x3 = GRU(64, return_sequences=True, name='key_sparse')(key_input)

    # x = merge([x1, x2, x3], mode='concat')
    x = merge([x1, x2], mode='concat')

    for i in range(layers):
        y = x
        x = GRU(
            num_units,
            return_sequences=i != layers - 1,
            name='lstm' + str(i)
        )(x)

        # Residual connection
        if i > 0 and i < layers - 1:
            x = merge([x, y], mode='sum')

        x = Activation('relu')(x)

        if dropout:
            x = Dropout(0.5)(x)

    return [note_input, beat_input], x


def note_model(time_steps):
    inputs, x = gru_stack(time_steps, False)

    # Multi-label
    policy = Dense(NUM_CLASSES, name='policy', activation='softmax')(x)
    value = Dense(1, name='value', activation='linear')(x)

    model = Model(inputs, [policy, value])
    model.load_weights('data/supervised.h5', by_name=True)
    # Create value output
    return model


def supervised_model(time_steps):
    inputs, x = gru_stack(time_steps)

    # Multi-label
    x = Dense(NUM_CLASSES, name='policy', activation='softmax')(x)

    model = Model(inputs, x)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def note_preprocess(env, x):
    note, beat = x
    return (one_hot(note, NUM_CLASSES), one_hot(beat, NOTES_PER_BAR))
