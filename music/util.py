import numpy as np
"""
Source:
https://github.com/tensorflow/magenta/blob/master/magenta/models/rl_tuner/rl_tuner_ops.py
"""
# The number of half-steps in musical intervals, in order of dissonance
OCTAVE = 12
FIFTH = 7
THIRD = 4
SIXTH = 9
SECOND = 2
FOURTH = 5
SEVENTH = 11
HALFSTEP = 1

# Note values of special actions.
NOTE_OFF = 0
NO_EVENT = 1

# Number of octaves supported
NUM_OCTAVES = 5

# Min note in MIDI supported
MIN_NOTE = 36
MAX_NOTE = MIN_NOTE + NUM_OCTAVES * OCTAVE

# Number of output note classes.
MIN_CLASS = 2  # First note class
NUM_CLASSES = MIN_CLASS + (MAX_NOTE - MIN_NOTE)

# Number of beats in a bar
BEATS_PER_BAR = 4
# Notes per quarter note
NOTES_PER_BEAT = 4
# The quickest note is a half-note
NOTES_PER_BAR = NOTES_PER_BEAT * BEATS_PER_BAR


# Music theory constants used in defining reward functions.
# Actions that are in C major
C_MAJOR_KEY = [0, 1]

# Only add octaves we want
for o in range(0, NUM_OCTAVES):
    C_MAJOR_KEY += [MIN_CLASS + o * OCTAVE + i for i in [0, 2, 4, 5, 7, 9, 11]]

C_MAJOR_TONIC = MIN_CLASS + MIN_NOTE + OCTAVE
A_MINOR_TONIC = C_MAJOR_TONIC + SIXTH

# Special intervals that have unique rewards
REST_INTERVAL = -1
HOLD_INTERVAL = -1.5
REST_INTERVAL_AFTER_THIRD_OR_FIFTH = -2
HOLD_INTERVAL_AFTER_THIRD_OR_FIFTH = -2.5
IN_KEY_THIRD = -3
IN_KEY_FIFTH = -5

# Indicate melody direction
ASCENDING = 1
DESCENDING = -1

# Indicate whether a melodic leap has been resolved or if another leap was made
LEAP_RESOLVED = 1
LEAP_DOUBLED = -1


def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,), dtype=int)
    arr[i] = 1
    return arr
