import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import load_model
from util import *
from midi_util import *
from music import NUM_CLASSES, NOTES_PER_BAR
from dataset import load_melodies, process_melody

melodies = list(map(process_melody, load_melodies(['data/edm'])))
# TODO: Harcode
style = [0.2, 0.8]
prev_styles = [style for _ in range(time_steps)]

with tf.device('/cpu:0'):
    samples = 5
    time_steps = 8
    BARS = 8

    model = load_model('data/supervised.h5')

    # Generate
    for sample_count in range(samples):
        # A priming melody
        inspiration = np.random.choice(melodies)

        # TODO: Refactor this with data set function calls
        prev_notes = deque(maxlen=time_steps)
        prev_beats = deque(maxlen=time_steps)

        i = NOTES_PER_BAR - 1
        for t in range(time_steps):
            # prev_notes.append(one_hot(inspiration[t], NUM_CLASSES))
            prev_notes.append(np.zeros(NUM_CLASSES))
            prev_beats.appendleft(one_hot(i, NOTES_PER_BAR))

            i -= 1
            if i < 0:
                i = NOTES_PER_BAR

        composition = []

        for i in range(NOTES_PER_BAR * BARS):
            results = model.predict([
                np.array([prev_notes]),
                np.array([prev_beats]),
                np.array([prev_styles])
            ])
            prob_dist = results[0]
            note = np.random.choice(len(prob_dist), p=prob_dist)

            result = one_hot(note, NUM_CLASSES)
            prev_notes.append(result)
            prev_beats.append(one_hot(i % NOTES_PER_BAR, NOTES_PER_BAR))
            composition.append(note)

        print('Composition', composition)
        mf = midi_encode_melody(composition)
        midi.write_midifile('out/melody_{}.mid'.format(sample_count), mf)
