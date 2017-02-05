# Trains an agent purely on music theory
import tensorflow as tf
from keras.models import load_model

from rl import A3CAgent, Memory
from util import *
from music import MusicCloneTunerEnv
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
    model_builder = lambda: MusicCloneTunerEnv(melodies, theory_scalar=1)

    agent.train(sess, model_builder).join()
