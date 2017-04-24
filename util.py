import os
import numpy as np

def one_hot(index, size):
    x = np.zeros(size)
    x[index] = 1
    return x

def get_all_files(paths, ext='.mid'):
    potential_files = []
    for path in paths:
        for root, dirs, files in os.walk(path):
            for f in files:
                fname = os.path.join(root, f)
                if os.path.isfile(fname) and fname.endswith(ext):
                    potential_files.append(fname)
    return potential_files
