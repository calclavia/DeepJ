import tensorflow as tf
from keras.layers import Input, LSTM, Dense, Lambda
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers.merge import Concatenate

from constants import SEQUENCE_LENGTH
from dataset import *
from music import OCTAVE

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

def build_model():
    notes_in = Input((SEQUENCE_LENGTH, NUM_NOTES))

    """ Time axis """
    # Pad note by one octave
    pad_note_layer = Lambda(lambda x: tf.pad(x, [[0, 0], [0, 0], [OCTAVE, OCTAVE]]), name='padded_note_in')
    padded_notes = pad_note_layer(notes_in)
    time_axis_rnn = LSTM(128, return_sequences=True, name='time_axis_rnn')
    time_axis_outs = []

    for n in range(OCTAVE, NUM_NOTES + OCTAVE):
        # Input one octave of notes
        octave_in = Lambda(lambda x: x[:, :, n - OCTAVE:n + OCTAVE + 1], name='note_' + str(n))(padded_notes)
        time_axis_outs.append(time_axis_rnn(octave_in))
    out = Concatenate()(time_axis_outs)

    """ Prediction Layer """
    predictions = Dense(NUM_NOTES, activation='sigmoid')(out)

    model = Model(notes_in, predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', f1_score])
    return model

def main():
    train()

def train():
    train_data, train_labels = load_all(['data/baroque'], BATCH_SIZE, SEQUENCE_LENGTH)

    try:
        model = load_model('out/model.h5')
        print('Loaded model from file.')
    except:
        model = build_model()
        print('Created new model.')

    model.summary()

    cbs = [ModelCheckpoint('out/model.h5', monitor='loss', save_best_only=True)]
    model.fit(train_data, train_labels, epochs=1000, callbacks=cbs)

if __name__ == '__main__':
    main()
