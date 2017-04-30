import tensorflow as tf
from keras.layers import Input, LSTM, Dense, Dropout, Lambda, Reshape, Permute, TimeDistributed
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, LambdaCallback, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.layers.merge import Concatenate, Add
from collections import deque
from tqdm import tqdm
import argparse

from constants import SEQUENCE_LENGTH
from dataset import *
from music import OCTAVE, NUM_OCTAVES
from midi_util import midi_encode
import midi

def build_model(time_steps=SEQUENCE_LENGTH, time_axis_units=256, note_axis_units=128, input_dropout=0.2, dropout=0.5):
    notes_in = Input((time_steps, NUM_NOTES))
    beat_in = Input((time_steps, NOTES_PER_BAR))
    # Target input for conditioning
    chosen_in = Input((time_steps, NUM_NOTES))

    """ Time axis """
    # Pad note by one octave
    notes = Dropout(input_dropout)(notes_in)
    padded_notes = Lambda(lambda x: tf.pad(x, [[0, 0], [0, 0], [OCTAVE, OCTAVE]]), name='padded_note_in')(notes)
    pitch_class_bins = Lambda(lambda x: tf.reduce_sum([x[:, :, i*OCTAVE:i*OCTAVE+OCTAVE] for i in range(NUM_OCTAVES)], axis=0), name='pitch_class_bins')(notes)

    time_axis_ins = []

    num_features = OCTAVE * 2 + 1 + 1 + OCTAVE + NOTES_PER_BAR
    # For every note...
    for n in range(OCTAVE, NUM_NOTES + OCTAVE):
        # Input one octave of notes
        octave_in = Lambda(lambda x: x[:, :, n - OCTAVE:n + OCTAVE + 1], name='note_' + str(n))(padded_notes)
        # Pitch position of note
        pitch_pos_in = Lambda(lambda x: tf.fill([tf.shape(x)[0], time_steps, 1], n / (NUM_NOTES - 1)))(notes_in)
        # Pitch class of current note
        pitch_class_in = Lambda(lambda x: tf.reshape(tf.tile(tf.constant(one_hot(n % OCTAVE, OCTAVE), dtype=tf.float32), [tf.shape(x)[0] * time_steps]), [tf.shape(x)[0], time_steps, OCTAVE]))(notes_in)

        time_axis_in = Concatenate()([octave_in, pitch_pos_in, pitch_class_in, beat_in])
        time_axis_ins.append(time_axis_in)

    # [batch, note, time, features]
    x = Lambda(lambda x: tf.stack(x, axis=1))(time_axis_ins)

    # Apply LSTMs
    x = TimeDistributed(LSTM(time_axis_units, return_sequences=True, activation='tanh'))(x)
    x = Dropout(dropout)(x)

    x = TimeDistributed(LSTM(time_axis_units, return_sequences=True, activation='tanh'))(x)
    x = Dropout(dropout)(x)

    # [batch, time, notes, features]
    out = Permute((2, 1, 3))(x)

    """ Note Axis & Prediction Layer """
    # Shift target one note to the left. []
    shift_chosen = Lambda(lambda x: tf.pad(x[:, :, :-1], [[0, 0], [0, 0], [1, 0]]))(chosen_in)
    shift_chosen = Dropout(input_dropout)(shift_chosen)
    shift_chosen = Lambda(lambda x: tf.expand_dims(x, -1))(shift_chosen)

    # [batch, time, notes, 1]
    shift_chosen = Reshape((time_steps, NUM_NOTES, -1))(shift_chosen)
    # [batch, time, notes, features + 1]
    note_axis_out = Concatenate(axis=3)([out, shift_chosen])

    note_axis_out = TimeDistributed(LSTM(note_axis_units, return_sequences=True, activation='tanh'))(note_axis_out)
    note_axis_out = Dropout(dropout)(note_axis_out)

    note_axis_out = TimeDistributed(LSTM(note_axis_units, return_sequences=True, activation='tanh'))(note_axis_out)
    note_axis_out = Dropout(dropout)(note_axis_out)

    note_axis_out = TimeDistributed(Dense(1, activation='sigmoid'))(note_axis_out)
    note_axis_out = Reshape((time_steps, NUM_NOTES))(note_axis_out)
    out = note_axis_out

    model = Model([notes_in, chosen_in, beat_in], out)
    model.compile(optimizer='nadam', loss='binary_crossentropy')
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

def build_or_load(allow_load=True):
    model = build_model()
    if allow_load:
        try:
            model.load_weights('out/model.h5')
            print('Loaded model from file.')
        except:
            print('Unable to load model from file.')
    model.summary()
    return model

def train(model, gen):
    print('Training')
    train_data, train_labels = load_all(styles, BATCH_SIZE, SEQUENCE_LENGTH)

    def epoch_cb(epoch, _):
        if epoch % 10 == 0:
            write_file(SAMPLES_DIR + '/result_epoch_{}.mid'.format(epoch), generate(model))

    cbs = [
        ModelCheckpoint('out/model.h5', monitor='loss', save_best_only=True),
        ReduceLROnPlateau(monitor='loss', patience=3),
        EarlyStopping(monitor='loss', patience=9),
        TensorBoard(log_dir='out/logs', histogram_freq=1)
    ]

    if gen:
        cbs += [LambdaCallback(on_epoch_end=epoch_cb)]

    model.fit(train_data, train_labels, epochs=1000, callbacks=cbs)

def generate(model, default_temp=1):
    print('Generating')
    notes_memory = deque([np.zeros(NUM_NOTES) for _ in range(SEQUENCE_LENGTH)], maxlen=SEQUENCE_LENGTH)
    beat_memory = deque([np.zeros(NOTES_PER_BAR) for _ in range(SEQUENCE_LENGTH)], maxlen=SEQUENCE_LENGTH)

    results = []
    temperature = default_temp

    for t in tqdm(range(NOTES_PER_BAR * 16)):

        # The next note being built.
        next_note = np.zeros(NUM_NOTES)

        # Generate each note individually
        for n in range(NUM_NOTES):
            predictions = model.predict([np.array([notes_memory]), np.array([list(notes_memory)[1:] + [next_note]]), np.array([beat_memory])])
            # We only care about the last time step
            prob_dist = predictions[0][-1]

            # Apply temperature
            if temperature != 1:
                # Inverse sigmoid
                x = -np.log(1 / np.array(prob_dist) - 1)
                # Apply temperature to sigmoid function
                prob_dist = 1 / (1 + np.exp(-x / temperature))

            # Flip on randomly
            next_note[n] = 1 if np.random.random() <= prob_dist[n] else 0

        # Increase temperature while silent.
        if np.count_nonzero(next_note) == 0:
            temperature += 0.05
        else:
            temperature = default_temp

        notes_memory.append(next_note)
        # Consistent with dataset representation
        beat_memory.append(compute_beat(t, NOTES_PER_BAR))
        results.append(next_note)

    return results

def write_file(name, results):
    mf = midi_encode(unclamp_midi(results))
    midi.write_midifile(name, mf)

if __name__ == '__main__':
    main()
