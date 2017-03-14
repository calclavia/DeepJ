import numpy as np
import tensorflow as tf
from music import NOTES_PER_BAR, MAX_NOTE, MIN_NOTE, NUM_OCTAVES, OCTAVE
from constants import NUM_STYLES
from util import one_hot
from tqdm import tqdm
from dataset import compute_beat, compute_completion

NUM_NOTES = MAX_NOTE - MIN_NOTE

class MusicModel:
    def __init__(self, batch_size, time_steps, training=True, dropout=0.5, activation=tf.nn.tanh, rnn_layers=1):
        input_dropout_keep_prob = 0.25 if training else 1
        dropout_keep_prob = 0.5 if training else 1

        self.init_states = []
        self.final_states = []

        def repeat(x):
            return np.reshape(np.repeat(x, batch_size * time_steps), [batch_size, time_steps, -1])

        def rnn(units):
            """
            Recurrent layer
            """
            def f(x):
                cell = tf.contrib.rnn.GRUCell(units, activation=activation)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
                # cell = tf.contrib.rnn.MultiRNNCell([cell] * rnn_layers)
                # Initial state of the memory.
                init_state = cell.zero_state(batch_size, tf.float32)
                rnn_out, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state)
                self.init_states.append(init_state)
                self.final_states.append(final_state)
                return rnn_out
            return f

        def rnn_conv(name, units, filter_size, stride=1, include_note_pitch=False):
            """
            Recurrent convolution Layer.
            Given a tensor of shape [batch_size, time_steps, features, channels],
            outputs a tensor of shape [batch_size, time_steps, features, channels]
            """
            def f(x, contexts):
                num_features = int(x.get_shape()[2])
                # num_channels = int(x.get_shape()[3])

                outs = []

                if num_features % stride != 0:
                    print('Warning! Stride not divisible.', num_features, stride)

                print('Layer {}: units={} len={} filter={} stride={}'.format(name, units, num_features, filter_size, stride))

                # Convolve every channel independently
                for i in range(0, num_features, stride):
                    with tf.variable_scope(name, reuse=len(outs) > 0):
                        inv_input = [x[:, :, i:i+filter_size], contexts]

                        # Include the context of how high the current input is.
                        if include_note_pitch:
                            # Position of note and pitch class of note
                            inv_input += [
                                tf.constant(repeat(i / (num_features - 1)), dtype='float'),
                                tf.constant(repeat(one_hot(i % OCTAVE, OCTAVE)), dtype='float')
                            ]

                        inv_input = tf.concat(inv_input, 2)
                        outs.append(rnn(units)(inv_input))
                        if i + filter_size == num_features:
                            break
                out = tf.concat(outs, 2)
                # Perform max pooling
                # out = tf.nn.pool(out, [2], strides=[2], pooling_type='MAX', padding='VALID', data_format='NCW')
                assert out.get_shape()[0] == batch_size
                assert out.get_shape()[1] == time_steps
                return out
            return f

        def time_axis_block(name, num_units=256):
            """
            Recurrent convolution Layer.
            Given a tensor of shape [batch_size, time_steps, features, channels],
            outputs a tensor of shape [batch_size, time_steps, features, channels]
            """
            def f(x, contexts):
                outs = []

                pitch_class_bins = tf.reduce_sum([x[:, :, i*OCTAVE:i*OCTAVE+OCTAVE] for i in range(NUM_OCTAVES)], axis=0)
                print('Pitch class bins', pitch_class_bins)

                # Pad by one octave
                x = tf.pad(x, [[0, 0], [0, 0], [OCTAVE, OCTAVE]])
                print('Padded note input by octave', x)

                # Process every note independently
                for i in range(OCTAVE, NUM_NOTES + OCTAVE):
                    with tf.variable_scope(name, reuse=len(outs) > 0):
                        inv_input = tf.concat([
                            x[:, :, i - OCTAVE:i + OCTAVE + 1],
                            contexts,
                            # Position of note
                            tf.constant(repeat(i / (NUM_NOTES - 1)), dtype='float'),
                            # Pitch class of current note
                            tf.constant(repeat(one_hot(i % OCTAVE, OCTAVE)), dtype='float'),
                            pitch_class_bins
                        ], 2)

                        outs.append(rnn(num_units)(inv_input))

                # Stack all outputs into a new dimension
                out = tf.stack(outs, axis=2)

                print(name, out)
                assert out.get_shape()[0] == batch_size
                assert out.get_shape()[1] == time_steps
                assert out.get_shape()[2] == NUM_NOTES
                assert out.get_shape()[3] == num_units

                return out
            return f

        def note_axis_block(name, num_units=128):
            """
            The pitch block that conditions each note's generation on the
            previous note within one time step.
            """
            def f(x, target):
                """
                Parameters:
                    x - The output of the time axis layer.
                        [batch, time_steps, notes, features]
                    target - The target output for training.
                              [batch, time_steps, notes]
                """
                # TODO: Could try using non-recurrent network.
                num_time_steps = x.get_shape()[1]
                # Prevent being over dependent upon wrong note
                target = tf.nn.dropout(target, input_dropout_keep_prob)

                outs = []

                # Every time slice has a note-axis RNN
                for t in range(num_time_steps):
                    # [batch, notes, features]
                    input_for_time = x[:, t, :, :]
                    # [batch, notes, 1]
                    target_for_time = tf.expand_dims(target[:, t, :], -1)
                    # Shift target vector for prediction
                    target_for_time = tf.pad(target_for_time, [[0, 0], [1, 0], [0, 0]])
                    # Remove last note
                    target_for_time = target_for_time[:, :-1, :]

                    assert target_for_time.get_shape()[0] == input_for_time.get_shape()[0]
                    assert target_for_time.get_shape()[1] == NUM_NOTES
                    assert target_for_time.get_shape()[2] == 1

                    rnn_input = tf.concat([
                        # Features for each note
                        input_for_time,
                        # Conditioned on the previously generated note
                        target_for_time
                    ], 2)

                    with tf.variable_scope(name, reuse=len(outs) > 0):
                        rnn_out = rnn(num_units)(rnn_input)

                        # Dense prediction layer
                        rnn_out = tf.layers.dense(inputs=rnn_out, units=1)
                        rnn_out = tf.squeeze(rnn_out, axis=[2])
                        outs.append(rnn_out)

                # Merge note-axis outputs for each time step.
                out = tf.stack(outs, axis=1)

                print(name, out)
                assert out.get_shape()[0] == batch_size
                assert out.get_shape()[1] == time_steps
                assert out.get_shape()[2] == NUM_NOTES

                return out
            return f

        """
        Input
        """
        # Input note (multi-hot vector)
        note_in = tf.placeholder(tf.float32, [batch_size, time_steps, NUM_NOTES], name='note_in')
        # Input beat (clock representation)
        beat_in = tf.placeholder(tf.float32, [batch_size, time_steps, NOTES_PER_BAR], name='beat_in')
        # Input progress (scalar representation)
        progress_in = tf.placeholder(tf.float32, [batch_size, time_steps, 1], name='progress_in')
        # Style bias (one-hot representation)
        style_in = tf.placeholder(tf.float32, [batch_size, time_steps, NUM_STYLES], name='style_in')

        # Target note to predict
        note_target = tf.placeholder(tf.float32, [batch_size, time_steps, NUM_NOTES], name='target_in')

        # Context to help generation
        contexts = tf.concat([beat_in, progress_in, style_in], 2)

        # Note input
        out = note_in
        out = tf.nn.dropout(out, input_dropout_keep_prob)
        print('note_in', out)

        out = time_axis_block('time_axis_block')(out, contexts)
        out = note_axis_block('note_axis_block')(out, note_target)

        """
        Sigmoid Layer
        """
        # Next note predictions
        logits = out
        self.prob = tf.nn.sigmoid(logits)
        # Classification prediction for f1 score
        self.pred = tf.round(self.prob)

        # Current global step we are on
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        """
        Loss
        """
        total_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=note_target))
        train_step = tf.train.AdamOptimizer().minimize(total_loss, global_step=self.global_step)

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
        with tf.device('/cpu:0'):
            self.saver = tf.train.Saver()

        """
        Statistics
        """
        self.build_summary(self.pred, note_target)

    def build_summary(self, predicted, actual):
        # F1 score statistic
        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.count_nonzero(predicted * actual)
        tn = tf.count_nonzero((predicted - 1) * (actual - 1))
        fp = tf.count_nonzero(predicted * (actual - 1))
        fn = tf.count_nonzero((predicted - 1) * actual)

        # Calculate accuracy, precision, recall and F1 score.
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        self.fmeasure = (2 * precision * recall) / (precision + recall)

        # Add metrics to TensorBoard.
        tf.summary.scalar('Accuracy', accuracy)
        tf.summary.scalar('Precision', precision)
        tf.summary.scalar('Recall', recall)
        tf.summary.scalar('f-measure', self.fmeasure)

        tf.summary.scalar('loss', self.loss)

        self.merged_summaries = tf.summary.merge_all()

    def train(self, sess, train_seqs, num_epochs, model_file, verbose=True):
        writer = tf.summary.FileWriter('out/summary', sess.graph, flush_secs=3)

        for epoch in range(num_epochs):
            # Shuffle sequence orders.
            order = np.random.permutation(len(train_seqs))
            t = tqdm(order)
            t.set_description('{}/{}'.format(epoch + 1, num_epochs))

            # Train every single sequence
            for i in t:
                seq = train_seqs[i]
                # Reset state every sequence
                states = [None for _ in self.init_states]

                tt = tqdm(seq)
                for note_in, beat_in, progress_in, style_in, label in tt:
                    # Build feed-dict
                    feed_dict = {
                        self.note_in: note_in,
                        self.beat_in: beat_in,
                        self.progress_in: progress_in,
                        self.style_in: style_in,
                        self.note_target: label
                    }

                    for tf_s, s in zip(self.init_states, states):
                        if s is not None:
                            feed_dict[tf_s] = s

                    pred, summary, step, _, *states = sess.run([
                            self.pred,
                            self.merged_summaries,
                            self.global_step,
                            self.train_step,
                        ] + self.final_states,
                        feed_dict
                    )

                    # Add summary to Tensorboard
                    writer.add_summary(summary, step)

                    if step % 100 == 0:
                        self.saver.save(sess, model_file, global_step=step)

                    tt.set_description('Step {}'.format(step))

        # Save the last epoch
        self.saver.save(sess, model_file)

    def generate(self, sess, style, inspiration=None, length=NOTES_PER_BAR * 8):
        total_len = length + (len(inspiration) if inspiration is not None else 0)
        # Resulting generation
        results = []
        # Reset state
        states = [None for _ in self.init_states]

        # Current note
        prev_note = np.zeros(NUM_NOTES)

        for i in tqdm(range(total_len)):
            current_beat = compute_beat(i, NOTES_PER_BAR)
            current_progress = compute_completion(i, total_len)

            # The next note being built.
            next_note = np.zeros(NUM_NOTES)

            for n in range(NUM_NOTES):
                # Build feed dict
                feed_dict = {
                    self.note_in: [[prev_note]],
                    self.beat_in: [[current_beat]],
                    self.progress_in: [[current_progress]],
                    self.style_in: [[style]],
                    self.note_target: [[next_note]]
                }

                for tf_s, s in zip(self.init_states, states):
                    if s is not None:
                        feed_dict[tf_s] = s

                prob, *next_states = sess.run([self.prob] + self.final_states, feed_dict)

                if inspiration is not None and i < len(inspiration):
                    # Priming notes
                    next_note = inspiration[i]
                    break
                else:
                    # Choose one probability at a time.
                    prob = prob[0][0]
                    next_note[n] = 1 if np.random.random() <= prob[n] else 0

            results.append(next_note)
            # Only advance state after one note has been predicted.
            states = next_states
            prev_note = next_note
        return results
