import numpy as np
from collections import deque
from keras.models import load_model
from util import *
from midi_util import *

time_steps = 8
BARS = 64

model = load_model('out/model.h5')

# Generate
prev_notes = deque(maxlen=time_steps)
prev_beats = deque(maxlen=time_steps)

i = BEATS_PER_BAR - 1
for _ in range(time_steps):
    prev_notes.append(np.zeros((NUM_NOTES,)))
    prev_beats.appendleft(one_hot(i, BEATS_PER_BAR))

    i -= 1
    if i < 0:
        i = BEATS_PER_BAR

composition = []

for i in range(BEATS_PER_BAR * BARS):
    results = model.predict([np.array([prev_notes]), np.array([prev_beats])])
    result = results[0]

    # Pick notes from probability distribution
    for index, p in enumerate(result):
        if np.random.random() <= p:
            result[index] = 1
        else:
            result[index] = 0

    prev_notes.append(result)
    prev_beats.append(one_hot(i % BEATS_PER_BAR, BEATS_PER_BAR))
    composition.append(result)

mf = midi_encode(composition)
midi.write_midifile('out/output.mid', mf)
