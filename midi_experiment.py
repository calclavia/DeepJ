import numpy as np
from keras.layers import Dense, Input, Activation, Flatten
from keras.layers.recurrent import GRU
from keras.models import Model
from collections import deque
from midi_util import *
import midi

time_steps = 5

compositions = load_midi()

# convert an array of values into a dataset matrix


def create_dataset(data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back)]
        dataX.append(a)
        dataY.append(data[i + look_back])
    return dataX, dataY

data_set, label_set = [], []

for c in compositions:
    x, y = create_dataset(c, time_steps)
    data_set += x
    label_set += y

data_set, label_set = np.array(data_set), np.array(label_set)

# Multi-hot vector of each note
note_input = Input(shape=(time_steps, NUM_NOTES), name='note_input')

x = note_input

for i in range(1):
    x = GRU(64, return_sequences=True, name='lstm' + str(i))(x)
    x = Activation('relu')(x)

x = GRU(64, name='lstm_last')(x)
x = Activation('relu')(x)

# Multi-label
x = Dense(NUM_NOTES)(x)
x = Activation('sigmoid')(x)

model = Model([note_input], x)
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(data_set, label_set)

# Generate
prev = deque([np.zeros((NUM_NOTES,))
              for _ in range(time_steps)], maxlen=time_steps)
composition = []

for i in range(64):
    results = model.predict(np.array([prev]))
    result = results[0]
    # Pick notes probabilistically
    for index, p in enumerate(result):
        if np.random.random() <= p:
            result[index] = 1
        else:
            result[index] = 0

    prev.append(result)
    composition.append(result)

mf = midi_encode(composition)
midi.write_midifile('out/output.mid', mf)
