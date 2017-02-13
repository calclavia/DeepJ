"""
Preprocesses input MIDI files and converts it as a Numpy file.
"""
from preprocess import midi_io, melodies_lib
import os
import numpy as np
from music import *
from tqdm import tqdm
from collections import deque


def process_melody(melody):
    """
    Converts a melody data to be compatible with the neural network.
    """
    res = []
    for x in melody:
        if x >= MAX_NOTE:
            res.append(NO_EVENT)
        elif x >= 0:
            res.append(max(x - MIN_NOTE, MIN_CLASS))
        else:
            # Apply shift to NOTE_OFF and NO_EVENT
            res.append(abs(x) - 1)
    return res


def load_melody(fname):
    try:
        seq_pb = midi_io.midi_to_sequence_proto(fname)
        melody = melodies_lib.midi_file_to_melody(seq_pb, steps_per_quarter=NOTES_PER_BEAT)
        melody.squash(MIN_NOTE, MAX_NOTE, 0)
        return melody
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


def load_melodies(paths, limit=None):
    files = get_all_files(paths)

    if limit is not None:
        files = files[:limit]

    print('Loading melodies from {} files'.format(len(files)))
    res = [load_melody(f) for f in tqdm(files)]

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


def build_history_buffer(time_steps, num_classes, notes_in_bar, style_hot):
    history = deque(maxlen=time_steps)
    for i in range(time_steps):
        history.appendleft([
            np.zeros(num_classes),
            one_hot(notes_in_bar - 1 - i, notes_in_bar),
            style_hot
        ])
    return history


def dataset_generator(melody_styles, time_steps, num_classes, notes_in_bar):
    for s, style in enumerate(melody_styles):
        style_hot = one_hot(s, len(melody_styles))

        for melody in style:
            # Recurrent history
            history = build_history_buffer(time_steps, num_classes, notes_in_bar, style_hot)

            for beat, note in enumerate(melody):
                note_hot = one_hot(note, num_classes)
                beat_hot = one_hot(beat % notes_in_bar, notes_in_bar)

                # Wrap around
                next_note_index = 0 if beat + 1 >= len(melody) else beat + 1
                next_note_hot = one_hot(melody[next_note_index], num_classes)

                history.append([note_hot, beat_hot, style_hot])

                # Yield the current input with target
                # yield [list(hist_buff), beat_hot, style_hot], [next_note_hot]
                yield zip(*history), [next_note_hot]


def main():
    loaded_data = load_melodies_thread(['data/edm', 'data/clean_midi'])
    np.save('data/data_cache.npy', loaded_data)

if __name__ == '__main__':
    main()
