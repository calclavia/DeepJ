import tensorflow as tf
from keras.layers import Input, LSTM, Dense, Dropout, Lambda, Reshape
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.layers.merge import Concatenate
from collections import deque
from tqdm import tqdm
import argparse

from constants import SEQUENCE_LENGTH
from dataset import *
from music import OCTAVE
from midi_util import midi_encode
import midi


def f1_score(actual, predicted):
    # F1 score statistic
    # Count true positives, true negatives, false positives and false negatives.
    tp = tf.count_nonzero(predicted * actual, dtype=tf.float32)
    tn = tf.count_nonzero((predicted - 1) * (actual - 1), dtype=tf.float32)
    fp = tf.count_nonzero(predicted * (actual - 1), dtype=tf.float32)
    fn = tf.count_nonzero((predicted - 1) * actual, dtype=tf.float32)

    # Calculate accuracy, precision, recall and F1 score.
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    # Prevent divide by zero
    zero = tf.constant(0, dtype=tf.float32)
    precision = tf.cond(tf.not_equal(tp, 0), lambda: tp / (tp + fp), lambda: zero)
    recall = tf.cond(tf.not_equal(tp, 0), lambda: tp / (tp + fn), lambda: zero)
    pre_f = 2 * precision * recall
    fmeasure = tf.cond(tf.not_equal(pre_f, 0), lambda: pre_f / (precision + recall), lambda: zero)
    return fmeasure

def build_model(time_steps=SEQUENCE_LENGTH, time_axis_units=64, note_axis_units=64):
    notes_in = Input((time_steps, NUM_NOTES))
    # Target input for conditioning
    chosen_in = Input((time_steps, NUM_NOTES))

    """ Time axis """
    # Pad note by one octave
    out = Dropout(0.2)(notes_in)
    padded_notes = Lambda(lambda x: tf.pad(x, [[0, 0], [0, 0], [OCTAVE, OCTAVE]]), name='padded_note_in')(out)
    time_axis_rnn = LSTM(time_axis_units, return_sequences=True, activation='tanh', name='time_axis_rnn')
    time_axis_outs = []

    for n in range(OCTAVE, NUM_NOTES + OCTAVE):
        # Input one octave of notes
        octave_in = Lambda(lambda x: x[:, :, n - OCTAVE:n + OCTAVE + 1], name='note_' + str(n))(padded_notes)
        time_axis_out = time_axis_rnn(octave_in)
        time_axis_outs.append(time_axis_out)

    out = Concatenate()(time_axis_outs)
    out = Dropout(0.5)(out)

    """ Note Axis & Prediction Layer """
    # Shift target one note to the left. []
    shift_chosen = Lambda(lambda x: tf.pad(x[:, :, :-1], [[0, 0], [0, 0], [1, 0]]))(chosen_in)
    shift_chosen = Dropout(0.2)(shift_chosen)
    shift_chosen = Lambda(lambda x: tf.expand_dims(x, -1))(shift_chosen)
    note_axis_rnn = LSTM(note_axis_units, return_sequences=True, activation='tanh', name='note_axis_rnn')
    prediction_layer = Dense(1, activation='sigmoid')
    note_axis_outs = []

    # Reshape inputs
    # [batch, time, notes, features]
    out = Reshape((time_steps, NUM_NOTES, -1))(out)
    # [batch, time, notes, 1]
    shift_chosen = Reshape((time_steps, NUM_NOTES, -1))(shift_chosen)
    # [batch, time, notes, features + 1]
    note_axis_input = Concatenate(axis=3)([out, shift_chosen])

    for t in range(time_steps):
        # [batch, notes, features + 1]
        sliced = Lambda(lambda x: x[:, t, :, :], name='time_' + str(t))(note_axis_input)
        note_axis_out = note_axis_rnn(sliced)
        note_axis_out = Dropout(0.5)(note_axis_out)
        note_axis_out = prediction_layer(note_axis_out)
        note_axis_out = Reshape((NUM_NOTES,))(note_axis_out)
        note_axis_outs.append(note_axis_out)
    out = Lambda(lambda x: tf.stack(x, axis=1))(note_axis_outs)

    model = Model([notes_in, chosen_in], out)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--train', default=False, action='store_true', help='Train model?')
    parser.add_argument('--gen', default=False, action='store_true', help='Generate after each epoch?')
    args = parser.parse_args()

    model = build_or_load()

    if args.train:
        train(model, args.gen)
    else:
        write_file(SAMPLES_DIR + '/result.mid', generate(model))

def build_or_load():
    model = build_model()
    try:
        model.load_weights('out/model.h5')
        print('Loaded model from file.')
    except:
        print('Unable to load model from file.')
    model.summary()
    return model

def train(model, gen):
    print('Training')
    train_data, train_labels = load_all(['data/baroque'], BATCH_SIZE, SEQUENCE_LENGTH)

    def epoch_cb(epoch, _):
        if epoch % 10 == 0:
            write_file(SAMPLES_DIR + '/result_epoch_{}.mid'.format(epoch), generate(model))

    cbs = [ModelCheckpoint('out/model.h5', monitor='loss', save_best_only=True)]
    
    if gen:
        cbs += [LambdaCallback(on_epoch_end=epoch_cb)]

    model.fit([train_data, train_labels], train_labels, epochs=1000, callbacks=cbs)

def generate(model):
    print('Generating')
    notes_memory = deque([np.zeros(NUM_NOTES) for _ in range(SEQUENCE_LENGTH)], maxlen=SEQUENCE_LENGTH)
    results = []

    def make_batch(next_note):
        note_hist = list(notes_memory)
        return [np.array([note_hist]), np.array([note_hist[1:] + [next_note]])]

    for t in tqdm(range(NOTES_PER_BAR * 4)):
        # The next note being built.
        next_note = np.zeros(NUM_NOTES)

        # Generate each note individually
        for n in range(NUM_NOTES):
            predictions = model.predict(make_batch(next_note))
            # We only care about the last time step
            prob = predictions[0][-1]
            # Flip on randomly
            next_note[n] = 1 if np.random.random() <= prob[n] else 0

        notes_memory.append(next_note)
        results.append(next_note)

    return results

def write_file(name, results):
    mf = midi_encode(unclamp_midi(results))
    midi.write_midifile(name, mf)

if __name__ == '__main__':
    main()
