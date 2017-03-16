import os

# Define the musical styles
styles = ['data/baroque', 'data/classical', 'data/romantic']
#styles = ['data/edm', 'data/southern_rock', 'data/hard_rock']
# styles = ['data/edm']
NUM_STYLES = len(styles)

# Training parameters
BATCH_SIZE = 128
TIME_STEPS = 16

# Hyper Parameters
STYLE_UNITS = 32
TIME_AXIS_UNITS = [256, 256]
NOTE_AXIS_UNITS = [128, 128, 256, 256, 512, 512]

# Move file save location
model_file = 'out/saves/model'
model_dir = os.path.dirname(model_file)
