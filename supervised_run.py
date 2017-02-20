import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import load_model
from util import *
from midi_util import *
from music import NUM_CLASSES, NOTES_PER_BAR
from dataset import load_melodies, process_melody, compute_beat
import math
import argparse

parser = argparse.ArgumentParser(description='Generates music.')
parser.add_argument('model', metavar='M', type=str,
                    help='Path to the model file')
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
    model = load_model(args.model)
    prev_styles = [style for _ in range(time_steps)]

    # Generate
    for sample_count in range(samples):
        # A priming melody
        inspiration = None

        while inspiration is None or len(inspiration) < time_steps:
            inspiration = np.random.choice(inspirations)

        # Prime the RNN
        history = deque(maxlen=time_steps)
        # TODO: Not DRY
        i = NOTES_PER_BAR - 1
        for t in range(time_steps):
            history.appendleft([
                np.zeros(NUM_CLASSES),
                compute_beat(i, NOTES_PER_BAR),
                np.zeros(1),
                style
            ])

            i -= 1
            if i < 0:
                i = NOTES_PER_BAR - 1

        # Compose
        composition = []

        N = NOTES_PER_BAR * BARS
        for i in range(N):
            results = model.predict([np.array([x]) for x in zip(*history)])
            prob_dist = results[0]
            note = np.random.choice(len(prob_dist), p=prob_dist)

            note_hot = one_hot(note, NUM_CLASSES)
            beat_input = compute_beat(i, NOTES_PER_BAR)
            completion_input = np.array([i / (N - 1)])
            history.append([note_hot, beat_input, completion_input, style])

            composition.append(note)

        print('Composition', composition)
        mf = midi_encode_melody(composition)
        midi.write_midifile('out/melody_{}.mid'.format(sample_count), mf)
