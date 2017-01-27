"""
Preprocesses input MIDI files and converts it as a Numpy file.
"""
from preprocess import midi_io
from preprocess import melodies_lib
from joblib import Parallel, delayed
from util import process_melody
import os
import numpy as np

def load_melody(fname):
    try:
        seq_pb = midi_io.midi_to_sequence_proto(fname)
        melody = melodies_lib.midi_file_to_melody(seq_pb)
        return process_melody(melody)
    except Exception as e:
        # print(e)
        return None

def load_melodies(paths):
    potential_files = []
    for path in paths:
        for root, dirs, files in os.walk(path):
            for f in files:
                fname = os.path.join(root, f)
                if os.path.isfile(fname):
                    potential_files.append(fname)

    res = Parallel(n_jobs=4, verbose=5)(delayed(load_melody)(f) for f in potential_files)
    print(res)
    out = []
    skipped = 0
    for melody in res:
        if melody == None:
            skipped += 1
        else:
            out.append(melody)

    print('Loaded {} melodies (skipped {})'.format(len(out), skipped))
    return out

if __name__ == '__main__':
    loaded_data = load_melodies(['data/edm', 'data/clean_midi'])
    np.save('data/data_cache.npy', loaded_data)
