import sys

# Import modules
sys.path.append('../gym-music')

import tensorflow as tf

from rl import A3CAgent
from music import MusicTheoryEnv
from music import NUM_CLASSES
from models import rnn_model

with tf.device('/cpu:0'):
  agent = A3CAgent(NUM_CLASSES, lambda: rnn_model())
  agent.train('music-theory-v0')
