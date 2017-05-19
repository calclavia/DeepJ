import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.callbacks import EarlyStopping, TensorBoard
import argparse
import midi
import os

from constants import *
from dataset import *
from generate import *
from midi_util import midi_encode
from model import *

def main():
    models = build_or_load()
    train(models)

def train(models):
    print('Loading data')
    train_data, train_labels = load_all(styles, BATCH_SIZE, SEQ_LEN)

    cbs = [
        ModelCheckpoint(MODEL_FILE, monitor='loss', save_best_only=True, save_weights_only=True),
        EarlyStopping(monitor='loss', patience=5),
        TensorBoard(log_dir='out/logs', histogram_freq=1)
    ]

    print('Training')
    models[0].fit(train_data, train_labels, epochs=1000, callbacks=cbs, batch_size=BATCH_SIZE)

if __name__ == '__main__':
    main()
