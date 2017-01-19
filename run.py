import tensorflow as tf
import gym

from rl import A3CAgent, track
from util import *
from midi_util import *
from music import label_compositions
import midi

with tf.device('/cpu:0'), tf.Session() as sess:
    env = track(gym.make('music-v0'))
    agent = make_agent()
    agent.load(sess)
    agent.run(sess, env)

    print('Reward', env.total_reward)
    midi.write_midifile('out/output.mid', midi_encode(env.composition))
