from music import NOTES_PER_BAR, MAX_NOTE, MIN_NOTE
import os

# Define the musical styles
styles = ['data/baroque', 'data/classical', 'data/romantic']
NUM_STYLES = len(styles)

NUM_NOTES = MAX_NOTE - MIN_NOTE

# Training parameters
BATCH_SIZE = 32
SEQUENCE_LENGTH = 8 * NOTES_PER_BAR

# Hyper Parameters
STYLE_UNITS = 32
TIME_AXIS_UNITS = 256
NOTE_AXIS_UNITS = 128

TIME_AXIS_LAYERS = 2
NOTE_AXIS_LAYERS = 2

# Move file save location
model_file = 'out/saves/model'
model_dir = os.path.dirname(model_file)
SAMPLES_DIR = 'out'
