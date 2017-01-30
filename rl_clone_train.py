import tensorflow as tf

from rl import A3CAgent
from util import *
from music import NOTES_PER_BAR

with tf.Session() as sess, tf.device('/cpu:0'):
    agent = make_agent()
    agent.add_agent(CloneAgentRunner)

    try:
        agent.load(sess)
        print('Loading last saved session')
    except:
        print('Starting new session')

    agent.compile(sess)
    agent.train(sess, 'music-theory-v0').join()
