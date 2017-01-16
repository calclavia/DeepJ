from keras.layers import Dense, Input, merge, Activation, Flatten
from keras.layers.recurrent import GRU

from midi_util import *

data_set = load_midi()

print(len(data_set))
