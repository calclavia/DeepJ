### MIDI Parameters ###
MIDI_VELOCITY = 128
# Number of possible notes
NUM_NOTES = 128
# Number of time shift quantizations
TIME_QUANTIZATION = 64
# Maximum time shift in seconds
MAX_TIME_SHIFT = 1
# Number of velocity buns
VEL_QUANTIZATION = 32

NOTE_ON_OFFSET = 0
NOTE_OFF_OFFSET = NOTE_ON_OFFSET + NUM_NOTES
TIME_OFFSET = NOTE_OFF_OFFSET + NUM_NOTES
VEL_OFFSET = TIME_OFFSET + TIME_QUANTIZATION
NUM_ACTIONS = VEL_OFFSET + VEL_QUANTIZATION

# Trainin Parameters
BATCH_SIZE = 32
SEQ_LEN = 800
# The higher this parameter, the less overlap in sequence data
SEQ_SPLIT = SEQ_LEN // 2
# Maximum silence time in seconds
SILENT_LENGTH = 3

# Sampling schedule decay
SCHEDULE_RATE = 1e-4
MIN_SCHEDULE_PROB = 0.5

# Style
STYLES = ['data/baroque', 'data/classical', 'data/romantic']#, 'data/modern', 'data/jazz']
# STYLES = ['data/baroque', 'data/classical', 'data/romantic', 'data/modern', 'data/jazz']
# STYLES = ['data/baroque']
NUM_STYLES = len(STYLES)

# Paths
OUT_DIR = 'out'
CACHE_DIR = 'out/cache'
SAMPLES_DIR = 'out/samples'

settings = {
    'force_cpu': False
}