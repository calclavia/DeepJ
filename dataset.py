"""
Preprocesses input MIDI files and converts it as a Numpy file.
"""
from preprocess import midi_io, melodies_lib
import os
import numpy as np

def load_melody(fname):
    try:
        seq_pb = midi_io.midi_to_sequence_proto(fname)
        return melodies_lib.midi_file_to_melody(seq_pb)
    except Exception as e:
        # print(e)
        return None

def get_all_files(paths):
    potential_files = []
    for path in paths:
        for root, dirs, files in os.walk(path):
            for f in files:
                fname = os.path.join(root, f)
                if os.path.isfile(fname):
                    potential_files.append(fname)
    return potential_files

def load_melodies(paths):
    files = get_all_files(paths)
    print('Loading melodies from {} files'.format(len(files)))
    res = [load_melody(f) for f in files]

    out = []
    skipped = 0
    for melody in res:
        if melody == None:
            skipped += 1
        else:
            out.append(melody)

    print('Loaded {} melodies (skipped {})'.format(len(out), skipped))
    return out

def load_melodies_thread(paths):
    from joblib import Parallel, delayed
    files = get_all_files(paths)
    print('Loading melodies from {} files'.format(len(files)))
    res = Parallel(n_jobs=4, verbose=5)(delayed(load_melody)(f) for f in files)

    out = []
    skipped = 0
    for melody in res:
        if melody == None:
            skipped += 1
        else:
            out.append(melody)

    print('Loaded {} melodies (skipped {})'.format(len(out), skipped))
    return out

def main():
    loaded_data = load_melodies_thread(['data/edm', 'data/clean_midi'])
    np.save('data/data_cache.npy', loaded_data)

if __name__ == '__main__':
    main()
