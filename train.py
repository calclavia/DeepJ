import tensorflow as tf

from rl import A3CAgent
from util import *
from midi_util import *
from music import label_compositions

label_compositions += load_midi('data/classical_c')

with tf.device('/cpu:0'):
  agent = make_agent()
  agent.train('music-clone-v0')
