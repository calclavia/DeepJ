import sys

# Import modules
sys.path.append('../gym-music')

import tensorflow as tf
import gym

from rl import A3CAgent, track
from music import *
from models import *

from midiutil.MidiFile import MIDIFile


def env_to_midi(env):
    """
    Takes a composition environment and converts it to MIDI note array
    """
    track = 0
    channel = 0
    # In beats
    time = 0
    # In BPM
    tempo = 120 * BEATS_PER_BAR / 4
    # 0-127, as per the MIDI standard
    volume = 127

    # One track, defaults to format 1 (tempo track automatically created)
    mf = MIDIFile(1)
    mf.addTempo(track, time, tempo)

    i = 0
    while i < len(env.composition):
        action = env.composition[i]
        if action != NO_EVENT and action != NOTE_OFF:
            pitch = MIN_NOTE + action

            # Compute the duration in beats by checking forward
            duration = 1

            j = i + 1
            while j < len(env.composition) and env.composition[j] == NO_EVENT:
                duration += 1
                j += 1

            print(time, pitch, duration)
            mf.addNote(track, channel, pitch, time, duration, volume)

        time = time + 1
        i += 1

    return mf

with tf.device('/cpu:0'), tf.Session() as sess:
    env = track(gym.make('music-theory-v0'))
    agent = A3CAgent(NUM_CLASSES, lambda: rnn_model(),
                     preprocess=note_preprocess)
    agent.load(sess)
    agent.run(sess, env)

    mf = env_to_midi(env)

    with open('out/output.mid', 'wb') as outf:
        mf.writeFile(outf)

    print('Composition', env.composition)
    print('Reward', env.total_reward)
