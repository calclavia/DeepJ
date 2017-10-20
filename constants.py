# Music Parameters
OCTAVE = 12
NUM_OCTAVES = 4
NUM_NOTES = OCTAVE * NUM_OCTAVES

# Number of beats in a bar
BEATS_PER_BAR = 4
# Notes per quarter note
NOTES_PER_BEAT = 4
# The quickest note is a half-note
NOTES_PER_BAR = NOTES_PER_BEAT * BEATS_PER_BAR

# Min and max note (in MIDI note number)
MIN_NOTE = 36
MAX_NOTE = MIN_NOTE + NUM_OCTAVES * OCTAVE

# MIDI Resolution
DEFAULT_RES = 96
MIDI_MAX_NOTES = 128
MAX_VELOCITY = 127

# Trainin Parameters
BATCH_SIZE = 32
SEQ_LEN = 8 * NOTES_PER_BAR
SEQ_SPLIT = SEQ_LEN // 2

# Hyper Parameters
TIME_AXIS_UNITS = 300
NOTE_AXIS_UNITS = 100
TIME_AXIS_LAYERS = 2
NOTE_AXIS_LAYERS = 2

BEAT_UNITS = 16
VICINITY_UNITS = 16
CHORD_UNITS = 16
STYLE_UNITS = 32
NOTE_UNITS = 3

# Sampling schedule decay
SCHEDULE_RATE = 0#2e-5
MIN_SCHEDULE_PROB = 0.6

# Style
# STYLES = ['data/baroque', 'data/classical', 'data/romantic', 'data/modern', 'data/jazz']
STYLES = ['data/baroque_single']
NUM_STYLES = len(STYLES)

# Paths
OUT_DIR = 'out'
CACHE_DIR = 'out/cache'
SAMPLES_DIR = 'out/samples'

settings = {
    'force_cpu': False
}

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

LSTM_UNITS = 256