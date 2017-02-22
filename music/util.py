import numpy as np
"""
Source:
https://github.com/tensorflow/magenta/blob/master/magenta/models/rl_tuner/rl_tuner_ops.py
"""
NUM_KEYS = 12

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
NUM_OCTAVES = 4

# Min and max note (in MIDI note number)
MIN_NOTE = 36
MAX_NOTE = MIN_NOTE + NUM_OCTAVES * OCTAVE

# Number of output note classes.
MIN_CLASS = 2  # First note class
NUM_CLASSES = MIN_CLASS + (MAX_NOTE - MIN_NOTE)

# Number of beats in a bar
BEATS_PER_BAR = 4
# Notes per quarter note
NOTES_PER_BEAT = 2
# The quickest note is a half-note
NOTES_PER_BAR = NOTES_PER_BEAT * BEATS_PER_BAR

# Music theory constants used in defining reward functions.
# Actions that are in C major
C_MAJOR_KEY = [0, 1]

# Only add octaves we want
for o in range(0, NUM_OCTAVES):
    C_MAJOR_KEY += [MIN_CLASS + o * OCTAVE + i for i in [0, 2, 4, 5, 7, 9, 11]]

C_MAJOR_TONIC = MIN_CLASS + OCTAVE * 2
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


def autocorrelate(signal, lag=1):
    """
    Gives the correlation coefficient for the signal's correlation with itself.
    Args:
      signal: The signal on which to compute the autocorrelation. Can be a list.
      lag: The offset at which to correlate the signal with itself. E.g. if lag
        is 1, will compute the correlation between the signal and itself 1 beat
        later.
    Returns:
      Correlation coefficient.
    """
    n = len(signal)
    x = np.asarray(signal) - np.mean(signal)
    c0 = np.var(signal)
    return (x[lag:] * x[:n - lag]).sum() / float(n) / c0


def similarity(li, sublist):
    """
    Return:
        An integer between 0 and len(sublist).
        Number of elements in the sublist equal to the main list from
        the tail of the list to the end.
    """
    assert type(li) is list
    assert type(sublist) is list

    if len(li) == 0:
        return 0

    # Sublist current index
    j = len(sublist) - 1
    max_len = 0

    for x in reversed(li):
        if x != sublist[j]:
            j = len(sublist) - 1

        if x == sublist[j]:
            j -= 1
            max_len = max((len(sublist) - 1) - j, max_len)
            # This is the max possible length. Break fast.
            if j == -1:
                return max_len

    return max_len

def is_sublist(li, sublist):
    assert type(li) is list
    assert type(sublist) is list

    if len(li) == 0:
        return None

    # Sublist current index
    j = 0

    for i, x in enumerate(li):
        if x == sublist[j]:
            j += 1

            if j == len(sublist):
                return i - (j - 1)
        else:
            j = 0
    return None
