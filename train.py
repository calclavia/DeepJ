import tensorflow as tf

from rl import A3CAgent
from util import *
from music import MusicTheoryEnv
from music import NUM_CLASSES

with tf.device('/cpu:0'):
  agent = make_agent()
  agent.train('music-theory-v0')
