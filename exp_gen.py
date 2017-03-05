import numpy as np
from collections import deque
from keras.models import load_model
from midi_util import *
from music import *
from dataset import get_all_files

NUM_NOTES = 128
time_steps = 16

model = load_model('out/model.h5')

model.summary()

# TODO: Seems like fails even when given det track
# TODO: Test force feed
# compositions = [load_midi(f) for f in get_all_files(['data/edm_c_chords'])]
compositions = [load_midi(f) for f in get_all_files(['data/classical/mozart_few'])]
comp = compositions[0]

# Generate
prev_notes = deque(maxlen=time_steps)
prev_beats = deque(maxlen=time_steps)

i = NOTES_PER_BAR - 1
for _ in range(time_steps):
    prev_notes.append(np.zeros((NUM_NOTES,)))
    prev_beats.append(one_hot(i, NOTES_PER_BAR))

    i -= 1
    if i < 0:
        i = NOTES_PER_BAR

composition = []

for i in range(4 * 4 * 8):
    results = model.predict([np.array([prev_notes]), np.array([prev_beats])])
    result = results[0]

    # Pick notes from probability distribution
    for index, p in enumerate(result):
        result[index] = 1 if np.random.random() <= p else 0
        # result[index] = 1 if p >= 0.5 else 0

    prev_notes.append(result)
    # prev_notes.append(comp[i])
    prev_beats.append(one_hot(i % NOTES_PER_BAR, NOTES_PER_BAR))
    composition.append(result)

mf = midi_encode(composition)
midi.write_midifile('out/output.mid', mf)
