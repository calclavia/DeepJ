# Trains an agent purely on music theory
import tensorflow as tf
import gym
from rl import A3CAgent
from util import *

with tf.Session() as sess, tf.device('/cpu:0'):
    agent = make_agent()

    try:
        agent.load(sess)
        print('Loading last saved session')
    except:
        agent.compile(sess)
        print('Starting new session')

    agent.train(sess, lambda: gym.make('music-theory-v0')).join()
