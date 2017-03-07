import numpy as np
from collections import deque
from keras.models import load_model
from midi_util import *
from music import *
from dataset import get_all_files
from exp_train import build_model, NUM_NOTES, time_steps, model_file, pos_context_const, pitch_context_const

# model = build_model()
# model.load_weights(model_file)
model = load_model(model_file)
model.summary()

# Generate
for s in range(5):
    print('Generating sample {}'.format(s))
    history = deque(maxlen=time_steps)

    for _ in range(time_steps):
        history.append([np.zeros(NUM_NOTES), np.zeros(NOTES_PER_BAR)])

    def make_inputs():
        return [np.repeat(np.expand_dims(x, 0), 1, axis=0) for x in zip(*history)] +\
               [np.array([pos_context_const]), np.array([pos_context_const])]

    composition = []

    for i in range(4 * 4 * 16):
        results = model.predict(make_inputs())
        result = results[0]

        # Pick notes from probability distribution
        for index, p in enumerate(result):
            result[index] = 1 if np.random.random() <= p else 0
            # result[index] = 1 if p >= 0.5 else 0

        history.append([result, one_hot(i % NOTES_PER_BAR, NOTES_PER_BAR)])
        composition.append(result)

    # Shift notes back up
    composition = np.concatenate((np.zeros((len(composition), MIN_NOTE)), composition), axis=1)

    mf = midi_encode(composition)
    midi.write_midifile('out/output_{}.mid'.format(s), mf)
