import numpy as np
from collections import deque
from keras.models import load_model
from midi_util import *

time_steps = 8

model = load_model('out/model.h5')

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
