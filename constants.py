### MIDI Parameters ###
MIDI_VELOCITY = 128
# Number of possible notes
NUM_NOTES = 128
# Number of time steps per second
TIME_QUANTIZATION = 100
MAX_TIME_SHIFT = 100
# Number of velocity buns
VEL_QUANTIZATION = 32

NOTE_ON_OFFSET = 0
NOTE_OFF_OFFSET = NOTE_ON_OFFSET + NUM_NOTES
TIME_OFFSET = NOTE_OFF_OFFSET + NUM_NOTES
VEL_OFFSET = TIME_OFFSET + TIME_QUANTIZATION
NUM_ACTIONS = VEL_OFFSET + VEL_QUANTIZATION

# Trainin Parameters
BATCH_SIZE = 64
SEQ_LEN = 1000
# The higher this parameter, the less overlap in sequence data
SEQ_SPLIT = SEQ_LEN // 5

# Hyper Parameters
LSTM_UNITS = 512

# Sampling schedule decay
SCHEDULE_RATE = 0#2e-5
MIN_SCHEDULE_PROB = 0.6

# Style
# STYLES = ['data/baroque', 'data/classical', 'data/romantic', 'data/modern', 'data/jazz']
# STYLES = ['data/baroque_single']
STYLES = ['data/baroque']
NUM_STYLES = len(STYLES)

# Paths
OUT_DIR = 'out'
CACHE_DIR = 'out/cache'
SAMPLES_DIR = 'out/samples'

settings = {
    'force_cpu': False
}