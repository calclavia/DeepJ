import numpy as np
import tensorflow as tf

from music import *
from midi_util import *

NUM_NOTES = MAX_NOTE - MIN_NOTE
time_steps = 16
batch_size = 1

class Model:
    def __init__(self):
        state_size = 300

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
        # Initial state of the memory.
        init_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, note_in, initial_state=init_state)

        ### Sigmoid Layer ###
        with tf.variable_scope('predict'):
            W = tf.get_variable('W', [state_size, NUM_NOTES])
            b = tf.get_variable('b', [NUM_NOTES], initializer=tf.constant_initializer(0.0))

            # Reshape rnn_outputs (applying this layer to all timesteps)
            rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
            target_reshaped = tf.reshape(note_target, [-1, NUM_NOTES])

            logits = tf.matmul(rnn_outputs, W) + b

        """
        Loss
        """
        total_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, target_reshaped))
        train_step = tf.train.AdamOptimizer().minimize(total_loss)

        self.note_in = note_in
        self.note_target = note_target

        self.init_state = init_state
        self.final_state = final_state

        self.loss = total_loss
        self.train_step = train_step

    def train(self, data_it, num_epochs=100, verbose=True, save=False):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_losses = []

            for epoch in range(num_epochs):
                training_loss = 0
                t_state = None
                # TODO: Shuffle
                for step, (X, Y) in enumerate(data_it):
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
                if verbose:
                    print("Average training loss for epoch", epoch, ":", training_loss/step)
                training_losses.append(training_loss/step)

        return training_losses

def create_dataset(data, look_back):
    dataX, dataY = [], []
    for i in range(len(data) - look_back - 1):
        dataX.append(data[i:(i + look_back)])
        dataY.append(data[i + 1:(i + look_back + 1)])
    return dataX, dataY

def chunk(a, size):
    return np.swapaxes(np.split(np.array(a), size), 0, 1)

print('Building graph')
model = Model()


print('Preparing training data')

# Create training data
# Scale. 8 * 4 notes
sequence = [48, 50, 52, 53, 55, 57, 59, 60]
sequence = [one_hot(x - MIN_NOTE, NUM_NOTES) + one_hot(x - MIN_NOTE - 12, NUM_NOTES) for x in sequence]
sequence = [[x] * 4 for x in sequence]
sequence = [y for x in sequence for y in x]
w_seq = np.concatenate((np.zeros((len(sequence), MIN_NOTE)), sequence), axis=1)
midi.write_midifile('out/baseline.mid', midi_encode(w_seq))

train_data, label_data = create_dataset(sequence, time_steps)

# Chunk into batches
train_data = chunk(train_data, batch_size)
label_data = chunk(label_data, batch_size)

print('Training...')
model.train(zip(train_data, label_data), 100)
