import tensorflow as tf
import gym

from rl import A3CAgent, track
from util import *
from midi_util import *
import midi

with tf.device('/cpu:0'), tf.Session() as sess:
    env = track(gym.make('music-v0'))
    env.num_notes = 128
    agent = make_agent()
    agent.load(sess)
    agent.run(sess, env)
    
    mf = midi_encode_melody(env.composition)
    midi.write_midifile('out/output.mid', mf)
