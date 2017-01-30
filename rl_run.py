import tensorflow as tf
import gym

from rl import A3CAgent, track
from util import *
from midi_util import *
from music import MusicGenEnv
import midi
from dataset import load_melodies, process_melody

melodies = list(map(process_melody, load_melodies(['data/edm'])))
samples = 5

with tf.device('/cpu:0'), tf.Session() as sess:
    agent = make_agent()
    agent.load(sess)

    for sample_count in range(samples):
        # A priming melody
        inspiration = np.random.choice(melodies)
        env = track(MusicGenEnv(inspiration))
        agent.run(sess, env)
        print('Composition', env.composition)
        mf = midi_encode_melody(env.composition)
        midi.write_midifile('out/tune_{}.mid'.format(sample_count), mf)
