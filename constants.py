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
BATCH_SIZE = 4
SEQ_LEN = 2 * NOTES_PER_BAR
TIME_AXIS_UNITS = 256
NOTE_AXIS_UNITS = 128

STYLES = ['data/baroque', 'data/classical', 'data/romantic']
NUM_STYLES = len(STYLES)

OUT_DIR = 'out'
CACHE_DIR = 'out/cache'
