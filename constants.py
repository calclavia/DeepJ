### MIDI Parameters ###
MIDI_VELOCITY = 128
# Number of possible notes
NUM_NOTES = 128
# Number of MIDI ticks per beat (PPQ)
TICKS_PER_BEAT = 96
# Number of velocity buns
VEL_QUANTIZATION = 32

TOKEN_IDS = [
    'note_on',      # Toggles a note on
    'note_off',     # Toggles a note off
    'note_inc',     # Moves note pointer up one note
    'note_dec',     # Moves note pointer down one note
    'wait',         # Wait one time step
]

# Trainin Parameters
SEQ_LEN = 1024 + 1
GRADIENT_CLIP = 10
SCALE_FACTOR = 2 ** 10
VOCAB_SIZE = 2048
# The number of train generator cycles per sequence
TRAIN_CYCLES = 1000
VAL_CYCLES = int(TRAIN_CYCLES * 0.05)

# Style
STYLES = ['data/baroque', 'data/classical', 'data/romantic', 'data/modern']
NUM_STYLES = len(STYLES)

# Paths
OUT_DIR = 'out'
CACHE_DIR = 'out/cache'
SAMPLES_DIR = 'out/samples'
# Synthesizer sound file
SOUND_FONT_PATH = CACHE_DIR + '/soundfont.sf2'
SOUND_FONT_URL = 'http://zenvoid.org/audio/acoustic_grand_piano_ydp_20080910.sf2'

settings = {
    'force_cpu': False
}
