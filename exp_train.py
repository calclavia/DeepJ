import numpy as np
import tensorflow as tf

from keras.layers import Dense, Input, Activation, Flatten, Dropout, merge, RepeatVector, Reshape, Permute, Lambda
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Convolution1D
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from midi_util import *
from music import *
from dataset import get_all_files
import midi
import os

NUM_NOTES = MAX_NOTE - MIN_NOTE
time_steps = 16
model_file = 'out/model.h5'

# Constant context
pos_context_const = np.arange(NUM_NOTES) / (NUM_NOTES - 1.)
pitch_context_const = np.mod(np.arange(NUM_NOTES), OCTAVE) / (OCTAVE - 1.)

# convert an array of values into a dataset matrix
def create_dataset(data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back)]
        dataX.append(a)
        dataY.append(data[i + look_back])
    return dataX, dataY

def create_beat_data(composition, beats_per_bar=NOTES_PER_BAR):
    """
    Augment the composition with the beat count in a bar it is in.
    """
    beat_patterns = []
    i = 0
    for note in composition:
        beat_pattern = np.zeros((beats_per_bar,))
        beat_pattern[i] = 1
        beat_patterns.append(beat_pattern)
        i = (i + 1) % beats_per_bar
    return beat_patterns

def repeat_const():
    def f(x):
        x = RepeatVector((time_steps))(x)
        x = Reshape((NUM_NOTES, time_steps, 1))(x)
        return x
    return f

def build_model(num_time_axis=2, act='relu'):
    # Multi-hot vector of each note
    note = Input(shape=(time_steps, NUM_NOTES), name='note')

    beat_input = Input(shape=(time_steps, NOTES_PER_BAR), name='beat_input')
    beat_context = Reshape((1, time_steps, NOTES_PER_BAR))(beat_input)
    beat_context = merge([beat_context for _ in range(NUM_NOTES)], mode='concat', concat_axis=1)

    # Positional context. How high is the current note?
    pos = Input(shape=(NUM_NOTES,), name='pos')
    pos_context = repeat_const()(pos)

    # A number from 0 to 12 indicating the class of the pitch
    pitch_class = Input(shape=(NUM_NOTES,), name='pitch_class')
    pitch_class_context = repeat_const()(pitch_class)

    out = note

    # Convolution layer for vicinity context
    out = TimeDistributed(Reshape((NUM_NOTES, 1)))(out)

    # Previous vicinity
    # TODO: Stacked conv may allow better feature extraction
    out = TimeDistributed(Convolution1D(1, 2 * OCTAVE + 1, border_mode='same'))(out)
    out = Activation(act)(out)
    out = Dropout(0.2)(out)

    out = TimeDistributed(Flatten())(out)

    # Time axis connections only (each note is "independent" of others)
    # Permute the input so the notes are in the temporal dimension, and
    # perform a hack on temporal slice
    out = Reshape((NUM_NOTES, time_steps, 1))(out)
    # Add context
    out = merge([out, pos_context, pitch_class_context, beat_context], mode='concat')

    out = TimeDistributed(GRU(200, return_sequences=True))(out)
    out = Activation(act)(out)
    out = Dropout(0.5)(out)

    out = TimeDistributed(GRU(200))(out)
    out = Activation(act)(out)
    out = Dropout(0.5)(out)

    """
    for i in range(4):
        out = Convolution1D(64 * (2 ** i), 3)(out)
        out = Activation(act)(out)
        out = MaxPooling1D()(out)
        out = Dropout(0.5)(out)
    
    out = Dense(256)(out)
    out = Activation(act)(out)
    out = Dropout(0.5)(out)
    """

    # Note axis connections
    # TODO: Recurrent connections resets might not make sense here.
    out = GRU(100, return_sequences=True)(out)
    out = Activation(act)(out)
    out = Dropout(0.5)(out)

    out = GRU(50, return_sequences=True)(out)
    out = Activation(act)(out)
    out = Dropout(0.5)(out)
    out = Flatten()(out)

    # Multi-label
    out = Dense(NUM_NOTES)(out)
    out = Activation('sigmoid')(out)

    model = Model([note, beat_input, pos, pitch_class], out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # compositions = [load_midi(f) for f in get_all_files(['data/classical/bach'])]
    compositions = [load_midi(f) for f in get_all_files(['data/classical/bach', 'data/classical/mozart', 'data/classical/beethoven'])]
    compositions = [m[:, MIN_NOTE:MAX_NOTE] for m in compositions]

    note_inputs = []
    note_targets = []
    beat_set = []

    for c in compositions:
        x, y = create_dataset(c, time_steps)
        note_inputs += x
        note_targets += y
        beat_set += create_dataset(create_beat_data(c), time_steps)[0]

    pos_inputs = np.array([pos_context_const for _ in note_inputs])
    pitch_class_inputs = np.array([pitch_context_const for _ in note_inputs])

    note_inputs = np.array(note_inputs)
    note_targets = np.array(note_targets)
    beat_set = np.array(beat_set)

    model = build_model()
    model.summary()

    cbs = [
        ModelCheckpoint(filepath=model_file, monitor='loss', save_best_only=True),
        ReduceLROnPlateau(monitor='loss', patience=3, verbose=1),
        EarlyStopping(monitor='loss', patience=10)
    ]

    model.fit(
        [note_inputs, beat_set, pos_inputs, pitch_class_inputs],
        note_targets,
        nb_epoch=1000,
        callbacks=cbs
    )
