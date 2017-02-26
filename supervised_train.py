import numpy as np
import midi
import os
import tensorflow as tf
import os.path
import random
import itertools
import argparse
from util import *
from music import NUM_CLASSES, NOTES_PER_BAR, MAX_NOTE, NO_EVENT
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from dataset import load_training_seq
from tqdm import tqdm
from models import *

time_steps = 8
nb_epochs = 1000

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--type', metavar='T', default='gru_stateful', type=str,
                        help='Type of model to use')
    parser.add_argument('--file', metavar='F', type=str,
                        default='out/supervised.h5',
                        help='Path to the model file')
    args = parser.parse_args()

    model = load_supervised_model(time_steps, args.file, globals()[args.type])
    train_stateful(model, args.file)

def train_stateless(model, model_file):
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

def train_stateful(model, model_file):
    sequences = load_training_seq(time_steps)
    # Keep track of lowest loss
    prev_loss = float('inf')
    no_improvements = 0
    patience = 3

    for epoch in itertools.count():
        print('Epoch {}:'.format(epoch))
        acc = 0
        loss = 0
        count = 0

        order = np.random.permutation(len(sequences))
        t = tqdm(order)
        for s in t:
            inputs, targets = sequences[s]
            for x, y in zip(inputs, targets):
                tr_loss, tr_acc = model.train_on_batch(x, y)
                
                acc += tr_acc
                loss += tr_loss
                count += 1
                t.set_postfix(loss=loss/count, acc=acc/count)
            model.reset_states()

        # TODO: Save model, stop early
        if loss < prev_loss:
            prev_loss = loss
            no_improvements = 0
            model.save(model_file)
        else:
            no_improvements += 1

        if no_improvements > patience:
            break

if __name__ == '__main__':
    main()
