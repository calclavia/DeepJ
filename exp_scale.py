import numpy as np
from keras.layers import Dense, Input, Activation, Flatten, Dropout, merge, RepeatVector, Reshape, Permute, Lambda
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Convolution1D
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from midi_util import *
from util import one_hot
from exp_train import create_dataset
from collections import deque
from music import *

"""
Results:
When trying to predict piano roll, unable to properly learn sequences at all.

- Experiment with cirriculum learning
- Dropout works better when more timesteps. Model is more robust to noise.
- Timesteps are the most important
- Tanh is way better for Softmax
- Batch norm had high accuracy quickly, but performed worse (with 3 layers)

t = 1. Unable to memorize full sequence, but partially.
t = 2. Unable to memorize full sequence. Better.
t = 4. 100% accuracy

Non-softmax is the cause of the problem.
"""

NUM_NOTES = MAX_NOTE - MIN_NOTE
time_steps = 16
num_units = 300
nb_filters = 1

# Create model
note_in = Input((time_steps, NUM_NOTES))
# One hot of the current position we're in
# pos_in = Input((time_steps, NUM_NOTES))
# output = merge([note_in, pos_in], mode='concat')

out = note_in

# Convolution layer for vicinity context
"""
out = Reshape((time_steps, NUM_NOTES, 1), name='conv_reshape')(out)
out = TimeDistributed(Convolution1D(nb_filters, 2 * OCTAVE + 1, border_mode='same', activation='tanh'), name='conv1')(out)
"""

# Time axis connections only (each note is "independent" of others)
# Permute the input so the notes are in the temporal dimension, and
# perform a hack on temporal slice
out = Reshape((NUM_NOTES, time_steps, nb_filters), name='rnn_reshape')(out)
# Add context
# out = merge([out, pos_context, pitch_class_context, beat_context], mode='concat')
out = TimeDistributed(GRU(num_units, activation='tanh'), name='rnn1')(out)
out = Dropout(0.5)(out)

out = TimeDistributed(Dense(1, activation='sigmoid'), name='dense')(out)
out = Flatten()(out)

model = Model(note_in, out)
# model = Model([note_in, pos_in], output)
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy', 'fbeta_score'])
model.summary()

# Create training data
# Scale. 8 * 4 notes
sequence = [48, 50, 52, 53, 55, 57, 59, 60]
sequence = [one_hot(x - MIN_NOTE, NUM_NOTES) + one_hot(x - MIN_NOTE - 12, NUM_NOTES) for x in sequence]
sequence = [[x] * 4 for x in sequence]
sequence = [y for x in sequence for y in x]
w_seq = np.concatenate((np.zeros((len(sequence), MIN_NOTE)), sequence), axis=1)
midi.write_midifile('out/baseline.mid', midi_encode(w_seq))

# TODO: Changing one note should not affect probability of note outside vicinity.

"""
note_seq = np.reshape(np.array(sequence).flatten('F'), [-1, 1])
pos_seq = np.reshape([one_hot(i, NUM_NOTES) for i in range(NUM_NOTES)] * len(sequence), [-1, NUM_NOTES])

note_train_data, label_data = create_dataset(note_seq, time_steps)
pos_train_data, _ = create_dataset(pos_seq, time_steps)
"""

note_train_data, label_data = create_dataset(sequence, time_steps)

cbs = [
    ReduceLROnPlateau(monitor='fbeta_score', patience=5, verbose=1),
    EarlyStopping(monitor='fbeta_score', patience=10)
]

# Shuffling has no impact
model.fit(
    # [np.array(note_train_data), np.array(pos_train_data)],
    np.array(note_train_data),
    np.array(label_data),
    nb_epoch=50,
    batch_size=1,
    callbacks=cbs
)

# Generate
for s in range(4):
    print('Generating: ', s)
    history = deque([[np.zeros(1), np.zeros(NUM_NOTES)]] * time_steps, maxlen=time_steps)
    composition = []

    for i in range(4 * 4 * 2):
        # Current piano roll slice
        note_roll = np.zeros(NUM_NOTES)

        # Predict one note at a time
        for n in range(NUM_NOTES):
            results = model.predict([np.repeat(np.expand_dims(x, 0), 1, axis=0) for x in zip(*history)])
            note_roll[n] = 1 if np.random.random() <= results[0] else 0

            if i % 2 == 0:
                # TODO: Predictive model
                history.append([[sequence[i][n]], one_hot(n, NUM_NOTES)])
            else:
                # TODO: Generative model
                history.append([[note_roll[n]], one_hot(n, NUM_NOTES)])

        composition.append(note_roll)

    composition = np.concatenate((np.zeros((len(composition), MIN_NOTE)), composition), axis=1)
    midi.write_midifile('out/result_{}.mid'.format(s), midi_encode(composition))
