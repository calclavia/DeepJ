import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm

from dataset import load_music_styles, get_all_files, compute_beat, compute_completion
from music import *
from midi_util import *
from util import chunk
from constants import NUM_STYLES, styles
from models import MusicModel

BATCH_SIZE = 32
TIME_STEPS = 16
model_file = 'out/saves/model'

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def stagger(data, time_steps):
    dataX, dataY = [], []

    # First note prediction
    data = [np.zeros_like(data[0])] + list(data)

    for i in range(len(data) - time_steps - 1):
        dataX.append(data[i:(i + time_steps)])
        dataY.append(data[i + 1:(i + time_steps + 1)])
    return dataX, dataY

def process(sequences):
    train_seqs = []

    for seq in sequences:
        train_data, label_data = stagger(seq, TIME_STEPS)

        beat_data = [compute_beat(i, NOTES_PER_BAR) for i in range(len(seq))]
        beat_data, _ = stagger(beat_data, TIME_STEPS)

        progress_data = [compute_completion(i, len(seq)) for i in range(len(seq))]
        progress_data, _ = stagger(progress_data, TIME_STEPS)

        # Chunk into batches
        train_data = chunk(train_data, BATCH_SIZE)
        beat_data = chunk(beat_data, BATCH_SIZE)
        progress_data = chunk(progress_data, BATCH_SIZE)
        label_data = chunk(label_data, BATCH_SIZE)
        train_seqs.append(list(zip(train_data, beat_data, progress_data, label_data)))
    return train_seqs

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--train', default=False, action='store_true', help='Train model?')
    parser.add_argument('--load', default=False, action='store_true', help='Load model?')
    args = parser.parse_args()

    print('Preparing training data')

    # Load training data
    # TODO: Cirriculum training. Increasing complexity. Increasing timestep details?
    # TODO: Random transpoe?
    # TODO: Random slices of subsequence?
    sequences = [load_midi(f) for f in get_all_files(['data/classical/bach'])]
    sequences = [np.minimum(np.ceil(m[:, MIN_NOTE:MAX_NOTE]), 1) for m in sequences]
    train_seqs = process(sequences)

    if args.train:
        with tf.Session() as sess:
            print('Training batch_size={} time_steps={}'.format(BATCH_SIZE, TIME_STEPS))
            train_model = MusicModel(BATCH_SIZE, TIME_STEPS)
            sess.run(tf.global_variables_initializer())
            if args.load:
                train_model.saver.restore(sess, model_file)
            else:
                sess.run(tf.global_variables_initializer())
            train_model.train(sess, train_seqs, 1000)

    reset_graph()

    with tf.Session() as sess:
        print('Generating...')
        gen_model = MusicModel(1, 1, training=False)
        gen_model.saver.restore(sess, model_file)

        for s in range(5):
            print('s={}'.format(s))
            composition = gen_model.generate(sess, np.random.choice(sequences)[:NOTES_PER_BAR])
            composition = np.concatenate((np.zeros((len(composition), MIN_NOTE)), composition), axis=1)
            midi.write_midifile('out/result_{}.mid'.format(s), midi_encode(composition))

if __name__ == '__main__':
    main()
