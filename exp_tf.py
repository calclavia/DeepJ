import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm

from dataset import get_all_files
from music import *
from midi_util import *

NUM_NOTES = MAX_NOTE - MIN_NOTE
BATCH_SIZE = 1
TIME_STEPS = 20
model_file = 'out/saves/model'

class Model:
    def __init__(self, batch_size=BATCH_SIZE, time_steps=TIME_STEPS):
        state_size = 300
        num_layers = 2
        global_dropout = 0.5
        """
        Input
        """
        # Input note of the current time step
        note_in = tf.placeholder(tf.float32, [batch_size, time_steps, NUM_NOTES])
        # Target note to predict
        note_target = tf.placeholder(tf.float32, [batch_size, time_steps, NUM_NOTES])

        """
        RNN
        """
        ### RNN Layer ###
        cell = tf.nn.rnn_cell.GRUCell(state_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=global_dropout)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=global_dropout)

        # Initial state of the memory.
        init_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, note_in, initial_state=init_state)

        ### Sigmoid Layer ###
        with tf.variable_scope('predict'):
            W = tf.get_variable('W', [state_size, NUM_NOTES])
            b = tf.get_variable('b', [NUM_NOTES], initializer=tf.constant_initializer(0.0))

            # Reshape rnn_outputs for training
            rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
            target_reshaped = tf.reshape(note_target, [-1, NUM_NOTES])
            logits = tf.matmul(rnn_outputs, W) + b

            # Next note predictions
            self.preds = tf.nn.sigmoid(logits)

        """
        Loss
        """
        total_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, target_reshaped))
        train_step = tf.train.AdamOptimizer().minimize(total_loss)

        """
        Set instance vars
        """
        self.note_in = note_in
        self.note_target = note_target

        self.init_state = init_state
        self.final_state = final_state

        self.loss = total_loss
        self.train_step = train_step

        # Saver
        self.saver = tf.train.Saver()

    def train(self, sess, data_it, num_epochs=100, verbose=True):
        for epoch in range(num_epochs):
            training_loss = 0
            t_state = None

            t = tqdm(data_it)

            for step, (X, Y) in enumerate(t):
                feed_dict = { self.note_in: X, self.note_target: Y }

                if t_state is not None:
                    feed_dict[self.init_state] = t_state

                t_loss, t_state, _ = sess.run([
                        self.loss,
                        self.final_state,
                        self.train_step
                    ],
                    feed_dict
                )

                training_loss += t_loss
                t.set_postfix(loss=training_loss/(step + 1))

            self.saver.save(sess, model_file)

    def generate(self, sess, length=NOTES_PER_BAR * 2):
        # Resulting generation
        results = []
        # Current RNN state
        state = None
        # Current note
        current_note = np.zeros(NUM_NOTES)

        for i in range(length):
            if state is not None:
                # Not the first prediction
                feed_dict = { self.note_in: [[current_note]], self.init_state: state }
            else:
                # First prediction
                feed_dict = { self.note_in: [[current_note]] }

            preds, state = sess.run([self.preds, self.final_state], feed_dict)
            preds = preds[0]
            # Randomly choose classes for each class
            current_note = np.zeros(NUM_NOTES)

            for n in range(NUM_NOTES):
                current_note[n] = 1 if np.random.random() <= preds[n] else 0

            results.append(current_note)
        return results

def create_dataset(data, look_back):
    dataX, dataY = [], []

    # First note prediction
    data = [np.zeros_like(data[0])] + list(data)

    for i in range(len(data) - look_back - 1):
        dataX.append(data[i:(i + look_back)])
        dataY.append(data[i + 1:(i + look_back + 1)])
    return dataX, dataY

def chunk(a, size):
    return np.swapaxes(np.split(np.array(a), size), 0, 1)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

parser = argparse.ArgumentParser(description='Generates music.')
parser.add_argument('--train', default=False, action='store_true', help='Train model?')
args = parser.parse_args()

print('Preparing training data')

# Create training data
# Scale. 8 * 4 notes
sequences = [load_midi(f) for f in get_all_files(['data/classical/bach'])]
sequences = [m[:, MIN_NOTE:MAX_NOTE] for m in sequences]

sequence = [48, 50, 52, 53, 55, 57, 59, 60]
sequence = [one_hot(x - MIN_NOTE, NUM_NOTES) + one_hot(x - MIN_NOTE - 12, NUM_NOTES) for x in sequence]
sequence = [[x] * 4 for x in sequence]
sequence = [y for x in sequence for y in x]

sequence = sequences[0]

w_seq = np.concatenate((np.zeros((len(sequence), MIN_NOTE)), sequence), axis=1)
midi.write_midifile('out/baseline.mid', midi_encode(w_seq))

train_data, label_data = create_dataset(sequence, TIME_STEPS)

# Chunk into batches
train_data = chunk(train_data, BATCH_SIZE)
label_data = chunk(label_data, BATCH_SIZE)

if args.train:
    with tf.Session() as sess:
        print('Training...')
        train_model = Model()
        sess.run(tf.global_variables_initializer())
        train_model.train(sess, list(zip(train_data, label_data)), 1)

reset_graph()

with tf.Session() as sess:
    print('Generating...')
    gen_model = Model(1, 1)
    gen_model.saver.restore(sess, model_file)

    for s in range(4):
        print('s={}'.format(s))
        composition = gen_model.generate(sess)
        composition = np.concatenate((np.zeros((len(composition), MIN_NOTE)), composition), axis=1)
        midi.write_midifile('out/result_{}.mid'.format(s), midi_encode(composition))
