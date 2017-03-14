import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm
import itertools

from dataset import load_styles, load_process_styles, unclamp_midi, clamp_midi
from music import *
from constants import styles, NUM_STYLES
from models import MusicModel
from midi_util import midi_encode
import midi

BATCH_SIZE = 64
TIME_STEPS = 16
# TODO: Model not restoring the latest file
model_file = 'out/saves/model-36800'

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--train', default=False, action='store_true', help='Train model?')
    parser.add_argument('--load', default=False, action='store_true', help='Load model?')
    args = parser.parse_args()

    print('Preparing training data')

    with tf.Session() as sess:
        # Load training data
        train_seqs = load_process_styles(styles, BATCH_SIZE, TIME_STEPS)

        if args.train:
            print('Training batch_size={} time_steps={}'.format(BATCH_SIZE, TIME_STEPS))
            train_model = MusicModel(BATCH_SIZE, TIME_STEPS)
            sess.run(tf.global_variables_initializer())

            if args.load:
                train_model.saver.restore(sess, model_file)
            else:
                sess.run(tf.global_variables_initializer())

            # TODO: Model not restoring the latest file
            train_model.train(sess, train_seqs, 1000, 'out/saves/model')
        else:
            print('Generating...')
            sequences = [clamp_midi(s) for s in load_styles(styles)]
            # All possible style enumerations
            all_styles = [np.array(i, dtype=float) for i in itertools.product([0, 1], repeat=NUM_STYLES)]

            gen_model = MusicModel(1, 1, training=False)
            gen_model.saver.restore(sess, model_file)

            for generate in range(5):
                for style in all_styles:
                    print('Sample: {} Style:'.format(generate, style))
                    composition = gen_model.generate(sess, style / np.sum(style), np.random.choice(sequences)[:NOTES_PER_BAR])
                    mf = midi_encode(unclamp_midi(composition))
                    midi.write_midifile('out/result_{}_{}.mid'.format(generate, style), mf)

if __name__ == '__main__':
    main()
