import numpy as np
import midi
import os
import tensorflow as tf
from util import *
from models import supervised_model
from music import NUM_CLASSES, NOTES_PER_BAR

from keras.callbacks import ModelCheckpoint

time_steps = 8
model_save_file = 'out/supervised.h5'

melodies = load_melodies('data/edm') + load_melodies('data/70s')

# TODO: Use memory
data_set, beat_set, label_set = [], [], []

for c in melodies:
    c = [one_hot(x, NUM_CLASSES) for x in c]
    x, y = create_dataset(c, time_steps)
    data_set += x
    label_set += y
    beat_data = create_beat_data(c, NOTES_PER_BAR)
    beat_set += create_dataset(beat_data, time_steps)[0]

data_set = np.array(data_set)
label_set = np.array(label_set)
beat_set = np.array(beat_set)

model = supervised_model(time_steps)

cbs = [ModelCheckpoint(filepath=model_save_file, monitor='loss', save_best_only=True)]

model.fit(
    [data_set, beat_set],
    label_set,
    nb_epoch=1000,
    callbacks=cbs
)
