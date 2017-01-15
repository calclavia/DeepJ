import sys

# Import modules
sys.path.append('../gym-music')

import tensorflow as tf

from rl import A3CAgent
from music import MusicTheoryEnv
from music import NUM_CLASSES
from util import *

with tf.device('/cpu:0'):
  agent = make_agent()
  agent.train('music-theory-v0')
