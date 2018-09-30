### MIDI Parameters ###
MIDI_VELOCITY = 128
# Number of possible notes
NUM_NOTES = 128

TICKS_PER_SEC = 100
# Number of velocity buns
VEL_QUANTIZATION = 32
TIME_QUANTIZATION = 100

TOKEN_EOS = 0
TOKEN_WAIT = 1
TOKEN_NOTE = TOKEN_WAIT + TIME_QUANTIZATION
TOKEN_VEL = TOKEN_NOTE + NUM_NOTES
NUM_TOKENS = TOKEN_VEL + VEL_QUANTIZATION
UNICODE_OFFSET = 0x4E00

# Trainin Parameters
SEQ_LEN = 1024 + 1
SCALE_FACTOR = 2 ** 10
VOCAB_SIZE = NUM_TOKENS
NUM_UNITS = 512
NUM_SEQS = 2048 # Maximum amount of sequences the model expects
STYLE_UNITS = 64

MAX_LR = 6
MIN_LR = MAX_LR / 20
MAX_ITER = 3000

RANDOM_TRANSPOSE = 4

# Paths
DATA_FOLDER = 'data'
OUT_DIR = 'out'
CACHE_DIR = 'out/cache'
SAMPLES_DIR = 'out/samples'
STYLE_NAMES = ['baroque', 'classical', 'romantic', 'modern']