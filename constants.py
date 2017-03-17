from music import NOTES_PER_BAR
import os

# Define the musical styles
styles = ['data/baroque', 'data/classical', 'data/romantic']
#styles = ['data/edm', 'data/southern_rock', 'data/hard_rock']
# styles = ['data/edm']
NUM_STYLES = len(styles)

# Training parameters
BATCH_SIZE = 32
SEQUENCE_LENGTH = NOTES_PER_BAR * 8
TIME_STEPS = SEQUENCE_LENGTH

# Hyper Parameters
STYLE_UNITS = 32
TIME_AXIS_UNITS = [256, 256]
NOTE_AXIS_UNITS = [128, 128, 128, 128, 128, 128]

# Move file save location
model_file = 'out/saves/model'
model_dir = os.path.dirname(model_file)
