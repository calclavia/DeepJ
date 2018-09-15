### MIDI Parameters ###
MIDI_VELOCITY = 128
# Number of possible notes
NUM_NOTES = 128

TICKS_PER_SEC = 50
# Number of velocity buns
VEL_QUANTIZATION = 32

# 0
TOKEN_WAIT = 0
# 1 -> 129
TOKEN_NOTE = 1
# 129 -> 161
TOKEN_VEL = TOKEN_NOTE + NUM_NOTES
NUM_TOKENS = TOKEN_VEL + VEL_QUANTIZATION
UNICODE_OFFSET = 0x4E00

# Trainin Parameters
SEQ_LEN = 1024 + 1
GRADIENT_CLIP = 10
SCALE_FACTOR = 2 ** 10
VOCAB_SIZE = 1024
# The number of train generator cycles per sequence
TRAIN_CYCLES = 1000
NUM_UNITS = 1024
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
