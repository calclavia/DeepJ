import numpy as np
import tensorflow as tf
import argparse
from sklearn import metrics
from tqdm import tqdm

from dataset import process_stateful, load_music_styles
from music import *
from midi_util import *
from constants import NUM_STYLES
from keras.layers.recurrent import GRU

NUM_NOTES = MAX_NOTE - MIN_NOTE + 2
BATCH_SIZE = 64
TIME_STEPS = 32
model_file = 'out/saves/model'

class Model:
    def __init__(self, batch_size=BATCH_SIZE, time_steps=TIME_STEPS, training=True, dropout=0.5):
        self.init_states = []
        self.final_states = []

        def rnn(units, out):
            cell = tf.contrib.rnn.GRUCell(units)
            # Initial state of the memory.
            init_state = cell.zero_state(batch_size, tf.float32)
            rnn_out, final_state = tf.nn.dynamic_rnn(cell, note_in, initial_state=init_state)
            rnn_out = tf.layers.dropout(inputs=rnn_out, rate=dropout, training=training)
            self.init_states.append(init_state)
            self.final_states.append(final_state)
            return rnn_out

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

        """
        Note invariant block
        """
        # Output of RNN for each note
        rnn_note_outs = []

        for i in range(NUM_NOTES):
            with tf.variable_scope('rnn1', reuse=i > 0):
                inv_input = tf.concat([note_in[:, :, i:i+1], beat_in, progress_in, style_in], 2)
                rnn_note_outs.append(rnn(128, inv_input))

        # Output of RNN for every octave
        rnn_octave_outs = []

        for i in range(NUM_OCTAVES):
            with tf.variable_scope('rnn2', reuse=i > 0):
                inv_input = tf.concat(rnn_note_outs[i:i+OCTAVE], 2)
                rnn_octave_outs.append(rnn(512, inv_input))

        out = tf.concat(rnn_octave_outs, 2)
        assert out.get_shape()[0] == batch_size
        assert out.get_shape()[1] == time_steps

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
        total_steps = 0
        patience = 3
        no_improvement = 0
        best_fscore = 0

        for epoch in range(num_epochs):
            # Metrics
            training_loss = 0
            f1_score = 0
            step = 0

            # Bar
            t = tqdm(train_seqs)
            t.set_description('{}/{}'.format(epoch + 1, num_epochs))

            for seq in t:
                # Reset state
                states = [None for _ in self.init_states]
                inputs, targets = seq

                for X, Y in tqdm(list(zip(inputs, targets))):
                    # Build feed-dict
                    feed_dict = {
                        self.note_in: X[0],
                        self.beat_in: X[1],
                        self.progress_in: X[2],
                        self.style_in: X[3],
                        self.note_target: Y
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
                    f1_score += np.mean([metrics.f1_score(y, p, average='weighted') for y, p in zip(Y, pred)])
                    t.set_postfix(loss=training_loss / step, f1=f1_score / step)

                    if total_steps % 1000 == 0:
                        # print('Saving at epoch {}'.format(epoch))
                        self.saver.save(sess, model_file)
                    total_steps += 1

            # Early stopping
            if f1_score > best_fscore:
                best_fscore = f1_score
                no_improvement = 0
            else:
                no_improvement += 1

                if no_improvement > patience:
                    break

        # Save the last epoch
        self.saver.save(sess, model_file)

    def generate(self, sess, inspiration, length=NOTES_PER_BAR * 16):
        # Resulting generation
        results = []
        # Reset state
        states = [None for _ in self.init_states]
        # Current note
        current_note = np.zeros(NUM_NOTES)

        for i in range(length + len(inspiration)):
            # Build feed dict
            feed_dict = { self.note_in: [[current_note]] }

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

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--train', default=False, action='store_true', help='Train model?')
    args = parser.parse_args()

    print('Preparing training data')

    # Load training data
    # TODO: Cirriculum training. Increasing complexity. Increasing timestep details?
    # TODO: Random transpoe?
    # TODO: Random slices of subsequence?
    sequences = process_stateful(load_music_styles(), TIME_STEPS, batch_size=BATCH_SIZE)

    if args.train:
        with tf.Session() as sess:
            print('Training...')
            train_model = Model()
            sess.run(tf.global_variables_initializer())
            train_model.train(sess, sequences, 100)

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
