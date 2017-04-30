import tensorflow as tf
from keras.layers import Input, LSTM, Dense, Dropout, Lambda, Reshape, Permute, TimeDistributed, RepeatVector, Conv1D
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, LambdaCallback, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.layers.merge import Concatenate, Add
from collections import deque
from tqdm import tqdm
import argparse

from constants import *
from dataset import *
from midi_util import midi_encode
import midi

def pitch_pos_in_f(time_steps):
    """
    Returns a constant containing pitch position of each note
    """
    def f(x):
        note_ranges = tf.range(NUM_NOTES, dtype='float32') / NUM_NOTES
        repeated_ranges = tf.tile(note_ranges, [tf.shape(x)[0] * time_steps])
        return tf.reshape(repeated_ranges, [tf.shape(x)[0], time_steps, NUM_NOTES, 1])
    return f

def pitch_class_in_f(time_steps):
    """
    Returns a constant containing pitch class of each note
    """
    def f(x):
        pitch_class_matrix = np.array([one_hot(n % OCTAVE, OCTAVE) for n in range(NUM_NOTES)])
        pitch_class_matrix = tf.constant(pitch_class_matrix, dtype='float32')
        pitch_class_matrix = tf.reshape(pitch_class_matrix, [1, 1, NUM_NOTES, OCTAVE])
        return tf.tile(pitch_class_matrix, [tf.shape(x)[0], time_steps, 1, 1])
    return f

def pitch_bins_f(time_steps):
    def f(x):
        bins = tf.reduce_sum([x[:, :, i::OCTAVE, 0] for i in range(OCTAVE)], axis=3)
        bins = tf.tile(bins, [NUM_OCTAVES, 1, 1])
        bins = tf.reshape(bins, [tf.shape(x)[0], time_steps, NUM_NOTES, 1])
        return bins
    return f

def build_model(time_steps=SEQUENCE_LENGTH, input_dropout=0.2, dropout=0.5):
    notes_in = Input((time_steps, NUM_NOTES))
    beat_in = Input((time_steps, NOTES_PER_BAR))
    # Target input for conditioning
    chosen_in = Input((time_steps, NUM_NOTES))

    # Dropout inputs
    notes = Dropout(input_dropout)(notes_in)
    beat = Dropout(input_dropout)(beat_in)
    chosen = Dropout(input_dropout)(chosen_in)

    # Reshape some inputs
    notes = Reshape((time_steps, NUM_NOTES, 1))(notes)
    chosen = Reshape((time_steps, NUM_NOTES, 1))(chosen)

    """ Time axis """
    # Create features for every single note.
    note_features = Concatenate()([
        Lambda(pitch_pos_in_f(time_steps))(notes),
        Lambda(pitch_class_in_f(time_steps))(notes),
        Lambda(pitch_bins_f(time_steps))(notes),
        # TODO: Don't hardcode
        TimeDistributed(Conv1D(32, 2 * OCTAVE, padding='same'))(notes),
        TimeDistributed(RepeatVector(NUM_NOTES))(beat)
    ])

    x = note_features

    # [batch, notes, time, features]
    x = Permute((2, 1, 3))(x)

    # Apply LSTMs
    for l in range(TIME_AXIS_LAYERS):
        x = TimeDistributed(LSTM(TIME_AXIS_UNITS, return_sequences=True))(x)
        x = Dropout(dropout)(x)

    # [batch, time, notes, features]
    x = Permute((2, 1, 3))(x)

    """ Note Axis & Prediction Layer """
    # Shift target one note to the left. []
    shift_chosen = Lambda(lambda x: tf.pad(x[:, :, :-1, :], [[0, 0], [0, 0], [1, 0], [0, 0]]))(chosen)

    # [batch, time, notes, 1]
    shift_chosen = Reshape((time_steps, NUM_NOTES, -1))(shift_chosen)
    # [batch, time, notes, features + 1]
    x = Concatenate(axis=3)([x, shift_chosen])

    for l in range(NOTE_AXIS_LAYERS):
        x = TimeDistributed(LSTM(NOTE_AXIS_UNITS, return_sequences=True))(x)
        x = Dropout(dropout)(x)

    x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    x = Reshape((time_steps, NUM_NOTES))(x)

    model = Model([notes_in, chosen_in, beat_in], x)
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

    for t in tqdm(range(NOTES_PER_BAR * 32)):

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
