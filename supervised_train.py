import numpy as np
import midi
import os
import tensorflow as tf
import os.path
import random
import itertools
from util import *
from music import NUM_CLASSES, NOTES_PER_BAR, MAX_NOTE, NO_EVENT
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from dataset import load_training_seq
from tqdm import tqdm

time_steps = 1
model_file = 'out/supervised.h5'
nb_epochs = 1000

def main():
    model = load_supervised_model(time_steps, model_file)
    sequences = load_training_seq()
    """
    input_set, target_set = load_training_data()

    cbs = [
        ModelCheckpoint(filepath=model_file, monitor='loss', save_best_only=True),
        #TensorBoard(log_dir='./out/supervised/summary', histogram_freq=1),
        ReduceLROnPlateau(monitor='loss', patience=5, verbose=1),
        EarlyStopping(monitor='loss', patience=10)
    ]

    model.fit(
        input_set,
        target_set,
        nb_epoch=1000,
        callbacks=cbs
    )
    """
    print('Training...')
    for epoch in itertools.count():
        print('Epoch {}:'.format(epoch))
        acc = 0
        loss = 0
        count = 0

        order = np.random.permutation(len(sequences))
        t = tqdm(order)
        for s in t:
            melody_inputs = sequences[s]
            for i in range(len(melody_inputs) - 1):
                tr_loss, tr_acc = model.train_on_batch(
                    melody_inputs[i],
                    np.reshape(melody_inputs[i + 1][0], [1, -1])
                )
                acc += tr_acc
                loss += tr_loss
                t.set_postfix(loss=loss/count, acc=acc/count)
                count += 1
            model.reset_states()

        # TODO: Save model, stop early

if __name__ == '__main__':
    main()
