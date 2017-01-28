# Trains an agent purely on music theory
import tensorflow as tf
from keras.models import load_model

from rl import A3CAgent, Memory
from util import *
from music import MusicTunerEnv
from models import note_preprocess

g_rnn = tf.Graph()
with g_rnn.as_default():
    supervised_model = load_model('data/supervised.h5')

with tf.Session() as sess, tf.device('/cpu:0'):
    agent = make_agent()

    try:
        agent.load(sess)
        print('Loading last saved session')
    except:
        print('Starting new session')

    agent.compile(sess)
    model_builder = lambda: MusicTunerEnv(
        g_rnn,
        supervised_model,
        Memory(8),
        preprocess=note_preprocess
    )

    agent.train(sess, model_builder).join()
