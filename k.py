import tensorflow as tf
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint

from constants import SEQUENCE_LENGTH
from dataset import *

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
    note_in = Input(shape=(SEQUENCE_LENGTH, NUM_NOTES))

    shared_lstm = LSTM(256)

    out = shared_lstm(note_in)

    predictions = Dense(NUM_NOTES, activation='sigmoid')(out)

    model = Model(inputs=note_in, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', f1_score])
    return model

def main():
    train()

def train():
    train_data, train_labels = load_all(['data/baroque'], BATCH_SIZE, SEQUENCE_LENGTH)

    try:
        model = load_model('out/model.h5')
    except:
        model = build_model()

    cbs = [ModelCheckpoint('out/model.h5', monitor='loss', save_best_only=True)]
    model.fit(train_data, train_labels, epochs=10, callbacks=cbs)

if __name__ == '__main__':
    main()
