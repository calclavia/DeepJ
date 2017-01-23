import tensorflow as tf

from rl import A3CAgent
from util import *
from midi_util import *

with tf.device('/cpu:0'):
  agent = make_agent()
  agent.train('music-theory-v0')
