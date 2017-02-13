import numpy as np
import midi
import os
import tensorflow as tf
import os.path
from util import *
from models import supervised_model
from music import NUM_CLASSES, NOTES_PER_BAR, MAX_NOTE, NO_EVENT
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from dataset import load_melodies, process_melody, dataset_generator

time_steps = 8
model_file = 'out/supervised.h5'

# Define the musical styles
styles = ['data/edm', 'data/country_rock']
NUM_STYLES = len(styles)

# A list of styles, each containing melodies
melody_styles = [list(map(process_melody, load_melodies([style]))) for style in styles]

print('Processing dataset')
input_set, target_set = zip(*dataset_generator(melody_styles, time_steps, NUM_CLASSES, NOTES_PER_BAR))
input_set = [np.array(i) for i in zip(*input_set)]
target_set = [np.array(i) for i in zip(*target_set)]

# Load model to continue training
if os.path.isfile(model_file):
    print('Loading model')
    model = load_model(model_file)
else:
    print('Creating new model')
    model = supervised_model(time_steps)

model.summary()

# Make dir for model saving
os.makedirs(os.path.dirname(model_file), exist_ok=True)

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
