import numpy as np
import tensorflow as tf
import math

from constants import *
from midi_util import *

def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,))
    arr[i] = 1
    return arr

def chunk(a, size):
    # Zero pad extra spaces
    target_size = math.ceil(len(a) / float(size)) * size
    inc_size = target_size - len(a)
    assert inc_size >= 0 and inc_size < size, inc_size
    a = np.array(a)
    a = np.pad(a, [(0, inc_size)] + [(0, 0) for i in range(len(a.shape) - 1)], mode='constant')
    assert a.shape[0] == target_size
    return np.swapaxes(np.split(a, size), 0, 1)

def get_all_files(paths):
    potential_files = []
    for path in paths:
        for root, dirs, files in os.walk(path):
            for f in files:
                fname = os.path.join(root, f)
                if os.path.isfile(fname) and fname.endswith('.mid'):
                    potential_files.append(fname)
    return potential_files
