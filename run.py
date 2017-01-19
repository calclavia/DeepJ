import tensorflow as tf
import gym

from rl import A3CAgent, track
from util import *
from music import *

with tf.device('/cpu:0'), tf.Session() as sess:
    env = track(gym.make('music-theory-v0'))
    env.num_notes = 128
    agent = make_agent()
    agent.load(sess)
    agent.run(sess, env)

    mf = env_to_midi(env)

    with open('out/output.mid', 'wb') as outf:
        mf.writeFile(outf)

    print('Composition', env.composition)
    print('Reward', env.total_reward)
