import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LambdaCallback, ReduceLROnPlateau
from keras.callbacks import EarlyStopping, TensorBoard
from collections import deque
from tqdm import tqdm
import argparse
import midi

from constants import *
from dataset import *
from midi_util import midi_encode
from model import *

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
            model.load_weights(MODEL_FILE)
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
        ModelCheckpoint(MODEL_FILE, monitor='loss', save_best_only=True),
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
