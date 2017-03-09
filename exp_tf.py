import numpy as np
import tensorflow as tf
import argparse
from sklearn import metrics
from tqdm import tqdm

from dataset import process_stateful, load_music_styles, get_all_files, compute_beat, compute_completion
from music import *
from midi_util import *
from util import chunk
from constants import NUM_STYLES, styles
from keras.layers.recurrent import GRU

NUM_NOTES = MAX_NOTE - MIN_NOTE
BATCH_SIZE = 64
TIME_STEPS = 32
model_file = 'out/saves/model'

class Model:
    def __init__(self, batch_size=BATCH_SIZE, time_steps=TIME_STEPS, training=True, dropout=0.5):
        self.init_states = []
        self.final_states = []

        def rnn(units):
            """
            Recurrent layer
            """
            def f(x):
                cell = tf.contrib.rnn.GRUCell(units)
                # Initial state of the memory.
                init_state = cell.zero_state(batch_size, tf.float32)
                rnn_out, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state)
                rnn_out = tf.layers.dropout(inputs=rnn_out, rate=dropout, training=training)
                self.init_states.append(init_state)
                self.final_states.append(final_state)
                return rnn_out
            return f

        def rnn_conv(name, units, filter_size, stride=1):
            """
            Recurrent convolution Layer
            """
            def f(x, contexts):
                total_len = int(x.get_shape()[2])
                outs = []
                assert total_len % stride == 0, (total_len, stride)
                print(name, 'units =', units, 'filter_size =', filter_size, 'stride =', stride)
                for i in range(0, total_len, stride):
                    with tf.variable_scope(name, reuse=i > 0):
                        inv_input = tf.concat([x[:, :, i:i+filter_size], contexts], 2)
                        outs.append(rnn(units)(inv_input))
                        if i + filter_size == total_len:
                            break
                out = tf.concat(outs, 2)
                # Perform max pooling
                out = tf.nn.pool(out, [2], strides=[2], pooling_type='MAX', padding='VALID', data_format='NCW')
                assert out.get_shape()[0] == batch_size
                assert out.get_shape()[1] == time_steps
                return out
            return f

        """
        Input
        """
        # Input note (multi-hot vector)
        note_in = tf.placeholder(tf.float32, [batch_size, time_steps, NUM_NOTES], name='note_in')
        # Input beat (clock representation)
        beat_in = tf.placeholder(tf.float32, [batch_size, time_steps, 2], name='beat_in')
        # Input progress (scalar representation)
        progress_in = tf.placeholder(tf.float32, [batch_size, time_steps, 1], name='progress_in')
        # Style bias (one-hot representation)
        style_in = tf.placeholder(tf.float32, [batch_size, time_steps, NUM_STYLES], name='style_in')

        # Target note to predict
        note_target = tf.placeholder(tf.float32, [batch_size, time_steps, NUM_NOTES], name='target_in')

        # Context to help generation
        contexts = tf.concat([beat_in, progress_in], 2)

        # Note input
        out = note_in
        """
        Note invariant block
        """
        last_units = 1
        for i in range(2):
            units = 128
            filter_size = last_units * (2 ** i)
            stride = last_units
            out = rnn_conv('rc' + str(i), units, filter_size, stride)(out, contexts)
            last_units = units
            print(out)

        """
        Sigmoid Layer
        """
        logits = tf.layers.dense(inputs=out, units=NUM_NOTES)

        # Next note predictions
        self.prob = tf.nn.sigmoid(logits)
        # Classification prediction for f1 score
        self.pred = tf.round(self.prob)

        """
        Loss
        """
        total_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=note_target))
        train_step = tf.train.AdamOptimizer().minimize(total_loss)

        """
        Set instance vars
        """
        self.note_in = note_in
        self.beat_in = beat_in
        self.progress_in = progress_in
        self.style_in = style_in
        self.note_target = note_target

        self.loss = total_loss
        self.train_step = train_step

        # Saver
        self.saver = tf.train.Saver()

    def train(self, sess, train_seqs, num_epochs=100, verbose=True):
        patience = 10
        no_improvement = 0
        best_fscore = 0

        for epoch in range(num_epochs):
            # Metrics
            training_loss = 0
            f1_score = 0
            step = 0

            # TODO: Shuffle sequence orders.
            # Bar
            t = tqdm(train_seqs)
            t.set_description('{}/{}'.format(epoch + 1, num_epochs))

            # Train every single sequence
            for seq in t:
                # Reset state
                states = [None for _ in self.init_states]

                for note_in, beat_in, progress_in, label in tqdm(seq):
                    # TODO: Dataset is bugged.
                    # Build feed-dict
                    feed_dict = {
                        self.note_in: note_in,
                        self.beat_in: beat_in,
                        self.progress_in: progress_in,
                        # self.style_in: X[3],
                        self.note_target: label
                    }

                    for tf_s, s in zip(self.init_states, states):
                        if s is not None:
                            feed_dict[tf_s] = s

                    pred, t_loss, _, *states = sess.run([
                            self.pred,
                            self.loss,
                            self.train_step
                        ] + self.final_states,
                        feed_dict
                    )

                    training_loss += t_loss
                    step += 1
                    # Compute F-1 score of all timesteps and batches
                    # For every single sample in the batch
                    f1_score += np.mean([metrics.f1_score(y, p, average='weighted') for y, p in zip(label, pred)])
                    t.set_postfix(loss=training_loss / step, f1=f1_score / step)

            # Early stopping
            if f1_score > best_fscore:
                self.saver.save(sess, model_file)
                best_fscore = f1_score
                no_improvement = 0
            else:
                no_improvement += 1

                if no_improvement > patience:
                    break

        # Save the last epoch
        self.saver.save(sess, model_file)

    def generate(self, sess, inspiration, length=NOTES_PER_BAR * 16):
        total_len = length + len(inspiration)
        # Resulting generation
        results = []
        # Reset state
        states = [None for _ in self.init_states]

        # Current note
        current_note = np.zeros(NUM_NOTES)
        current_beat = compute_beat(0, NOTES_PER_BAR)
        current_progress = compute_completion(0, total_len)

        for i in range(total_len):
            # Build feed dict
            feed_dict = {
                self.note_in: [[current_note]],
                self.beat_in: [[current_beat]],
                self.progress_in: [[current_progress]]
            }

            for tf_s, s in zip(self.init_states, states):
                if s is not None:
                    feed_dict[tf_s] = s

            prob, *states = sess.run([self.prob] + self.final_states, feed_dict)

            if i < len(inspiration):
                # Priming notes
                current_note = inspiration[i]
            else:
                prob = prob[0][0]
                # Randomly choose classes for each class
                current_note = np.zeros(NUM_NOTES)

                for n in range(NUM_NOTES):
                    current_note[n] = 1 if np.random.random() <= prob[n] else 0

                results.append(current_note)
        return results

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
    sequences = [load_midi(f) for f in get_all_files(styles)]
    sequences = [np.minimum(np.ceil(m[:, MIN_NOTE:MAX_NOTE]), 1) for m in sequences]
    train_seqs = process(sequences)

    if args.train:
        with tf.Session() as sess:
            print('Training...')
            train_model = Model()
            sess.run(tf.global_variables_initializer())
            if args.load:
                train_model.saver.restore(sess, model_file)
            else:
                sess.run(tf.global_variables_initializer())
            train_model.train(sess, train_seqs, 1000)

    reset_graph()

    with tf.Session() as sess:
        print('Generating...')
        gen_model = Model(1, 1, training=False)
        gen_model.saver.restore(sess, model_file)

        for s in range(5):
            print('s={}'.format(s))
            composition = gen_model.generate(sess, np.random.choice(sequences)[:NOTES_PER_BAR])
            composition = np.concatenate((np.zeros((len(composition), MIN_NOTE)), composition), axis=1)
            midi.write_midifile('out/result_{}.mid'.format(s), midi_encode(composition))

if __name__ == '__main__':
    main()
