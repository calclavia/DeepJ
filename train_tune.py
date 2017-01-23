# Trains an agent purely on music theory
import tensorflow as tf

from rl import A3CAgent
from util import *

with tf.Session() as sess, tf.device('/cpu:0'):
    agent = make_agent()

    try:
        agent.load(sess)
        print('Loading last saved session')
    except:
        print('Starting new session')

    agent.compile(sess)
    agent.train(sess, 'music-tuner-v0').join()
