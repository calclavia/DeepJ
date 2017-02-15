import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import load_model
from util import *
from midi_util import *
from music import NUM_CLASSES, NOTES_PER_BAR
from dataset import load_melodies, process_melody

import argparse

parser = argparse.ArgumentParser(description='Generates music.')
parser.add_argument('style', metavar='S', default=[], type=float, nargs='+',
                    help='A list that defines the weights of style')
parser.add_argument('--bars', default=8, type=int, dest='bars',
                    help='How many bars of music to generate.')

args = parser.parse_args()

style = args.style
samples = 5
time_steps = 10
BARS = args.bars

print('Generating music with style {} for {} bars'.format(style, BARS))
assert len(style) == NUM_STYLES
assert (1 - sum(style)) < 1e-2

# Inspiration melodies
inspirations = list(map(process_melody, load_melodies(styles)))

with tf.device('/cpu:0'):
    model = load_model('data/supervised.h5')
    prev_styles = [style for _ in range(time_steps)]

    # Generate
    for sample_count in range(samples):
        # A priming melody
        inspiration = None

        while inspiration is None or len(inspiration) < time_steps:
            inspiration = np.random.choice(inspirations)

        # TODO: Refactor this with data set function calls
        prev_notes = deque(maxlen=time_steps)
        prev_beats = deque(maxlen=time_steps)

        i = NOTES_PER_BAR - 1
        for t in range(time_steps):
            prev_notes.append(one_hot(inspiration[t], NUM_CLASSES))
            # prev_notes.append(np.zeros(NUM_CLASSES))
            prev_beats.appendleft(one_hot(i, NOTES_PER_BAR))

            i -= 1
            if i < 0:
                i = NOTES_PER_BAR - 1

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
