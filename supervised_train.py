import numpy as np
import midi
import os
import tensorflow as tf
import os.path
from util import *
from models import supervised_model
from music import NUM_CLASSES, NOTES_PER_BAR, MAX_NOTE, NO_EVENT
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from dataset import load_melodies, process_melody

time_steps = 8
model_file = 'out/supervised.h5'

melodies = list(map(process_melody, load_melodies(['data/edm', 'data/70s'])))

data_set, beat_set, label_set = [], [], []

for c in melodies:
    c_hot = [one_hot(x, NUM_CLASSES) for x in c]
    x, y = create_dataset(c_hot, time_steps)
    data_set += x
    label_set += y
    beat_data = create_beat_data(c, NOTES_PER_BAR)
    beat_set += create_dataset(beat_data, time_steps)[0]

data_set = np.array(data_set)
label_set = np.array(label_set)
beat_set = np.array(beat_set)

# Load model to continue training
if os.path.isfile(model_file):
    print('Loading model')
    model = load_model(model_file)
else:
    print('Creating new model')
    model = supervised_model(time_steps)

# Make dir for model saving
os.makedirs(os.path.dirname(model_file), exist_ok=True)

cbs = [ModelCheckpoint(filepath=model_file, monitor='loss', save_best_only=True)]

model.fit(
    [data_set, beat_set],
    label_set,
    nb_epoch=300,
    callbacks=cbs
)
