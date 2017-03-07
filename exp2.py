import numpy as np
from keras.layers import Dense, Input, Activation
from keras.layers.recurrent import GRU
from keras.models import Model
from midi_util import *
from util import one_hot

NUM_NOTES = 128
time_steps = 1

# Create model
note_in = Input((time_steps, NUM_NOTES))
output = note_in
output = GRU(100)(output)
output = Dense(NUM_NOTES)(output)
model = Model(note_in, output)

# Create training data
# C Major Scale
sequence = [48, 50, 52, 53, 55, 57, 59, 60]
sequence = [one_hot(x, NUM_NOTES) for x in sequence]
sequence = [[x] * 4 for x in sequence]
sequence = [y for x in sequence for y in x]

midi.write_midifile('out/baseline.mid', midi_encode(sequence))
