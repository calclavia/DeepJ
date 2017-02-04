import tensorflow as tf

from rl import A3CAgent
from util import *
from music import MusicCloneEnv
from dataset import load_melodies, process_melody

melodies = list(map(process_melody, load_melodies(['data/edm', 'data/70s'])))

with tf.Session() as sess, tf.device('/cpu:0'):
    agent = make_agent()

    try:
        agent.load(sess)
        print('Loading last saved session')
    except:
        print('Starting new session')

    agent.compile(sess)
    model_builder = lambda: MusicCloneEnv(melodies)

    agent.train(sess, model_builder).join()
