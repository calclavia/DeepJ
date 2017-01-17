import numpy as np
from keras.layers import Dense, Input, Activation, Flatten, Dropout, merge
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D
from keras.models import Model
from midi_util import *
import midi
import os

time_steps = 8
model_save_file = 'out/model.h5'

compositions = load_midi('data/edm_c_chords')

BEATS_PER_BAR = TICKS_PER_BEAT * BEATS_PER_BAR

def create_beat_data(composition, beats_per_bar=BEATS_PER_BAR):
    """
    Augment the composition with the beat count in a bar it is in.
    """
    beat_patterns = []
    i = 0
    for note in composition:
        beat_pattern = np.zeros((beats_per_bar,))
        beat_pattern[i] = 1
        beat_patterns.append(beat_pattern)
        i = (i + 1) % beats_per_bar
    return beat_patterns

# convert an array of values into a dataset matrix


def create_dataset(data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back)]
        dataX.append(a)
        dataY.append(data[i + look_back])
    return dataX, dataY

data_set, beat_set, label_set = [], [], []

for c in compositions:
    x, y = create_dataset(c, time_steps)
    data_set += x
    label_set += y
    beat_set += create_dataset(create_beat_data(c), time_steps)[0]

data_set = np.array(data_set)
label_set = np.array(label_set)
beat_set = np.array(beat_set)

# Multi-hot vector of each note
note_input = Input(shape=(time_steps, NUM_NOTES), name='note_input')
beat_input = Input(shape=(time_steps, BEATS_PER_BAR), name='beat_input')

x = note_input

x = Dropout(0.2)(x)
# Conv layer
for i in range(1):
    x = Convolution1D(64, 8, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

x = merge([x, beat_input], mode='concat')

for i in range(1):
    x = GRU(256, return_sequences=True, name='lstm' + str(i))(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

x = GRU(256, name='lstm_last')(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

for i in range(1):
    x = Dense(128, name='dense' + str(i))(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

# Multi-label
x = Dense(NUM_NOTES)(x)
x = Activation('sigmoid')(x)

model = Model([note_input, beat_input], x)
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit([data_set, beat_set], label_set, nb_epoch=500)

model.save(model_save_file)
