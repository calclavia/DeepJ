import numpy as np
import midi
import os
import tensorflow as tf
import os.path
from util import *
from music import NUM_CLASSES, NOTES_PER_BAR, MAX_NOTE, NO_EVENT
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from dataset import load_melodies, dataset_generator
from constants import NUM_STYLES, styles

time_steps = NOTES_PER_BAR
model_file = 'out/supervised.h5'

def load_data():
    # A list of styles, each containing melodies
    melody_styles = [load_melodies([style]) for style in styles]

    print('Processing dataset')
    input_set, target_set = zip(*dataset_generator(melody_styles, time_steps, NUM_CLASSES, NOTES_PER_BAR))
    input_set = [np.array(i) for i in zip(*input_set)]
    target_set = [np.array(i) for i in zip(*target_set)]
    return input_set, target_set

def main():
    model = load_supervised_model(time_steps, model_file)
    input_set, target_set = load_data()

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

if __name__ == '__main__':
    main()
