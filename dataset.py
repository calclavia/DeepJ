"""
Preprocesses input MIDI files and converts it as a Numpy file.
"""
from preprocess import midi_io, melodies_lib
import os
import numpy as np
from music import *
from tqdm import tqdm
from collections import deque
from joblib import Parallel, delayed
import math

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

def load_then_process(f):
    melody = load_melody(f)
    if melody:
        return process_melody(melody)
    else:
        return None

def load_melodies(paths, process=True, limit=None):
    assert len(paths) > 0
    files = get_all_files(paths)

    if limit is not None:
        files = files[:limit]

    print('Loading melodies from {} files'.format(len(files)))
    fn = load_then_process if process else load_melody
    res = Parallel(n_jobs=8, verbose=5, backend='threading')(delayed(fn)(f) for f in files)

    out = []
    skipped = 0
    for melody in res:
        if melody == None:
            skipped += 1
        else:
            out.append(melody)

    print('Loaded {} melodies (skipped {})'.format(len(out), skipped))
    return out



def compute_beat(beat, notes_in_bar):
    # Angle method
    angle = (beat % notes_in_bar) / notes_in_bar * 2 * math.pi
    return np.array([math.cos(angle), math.sin(angle)])

    #return one_hot(beat % notes_in_bar, notes_in_bar)

def build_history_buffer(time_steps, num_classes, notes_in_bar, style_hot):
    history = deque(maxlen=time_steps)
    current_beat = NOTES_PER_BAR - 1
    for i in range(time_steps):
        assert current_beat >= 0
        history.appendleft([
            np.zeros(num_classes),
            compute_beat(current_beat, notes_in_bar),
            np.zeros(1),
            style_hot
        ])

        current_beat -= 1
        if current_beat < 0:
            current_beat = NOTES_PER_BAR - 1
    return history


def dataset_generator(melody_styles,
                      time_steps,
                      num_classes,
                      notes_in_bar,
                      train_all=False):
    for s, style in enumerate(melody_styles):
        style_hot = one_hot(s, len(melody_styles))

        for melody in style:
            # Recurrent history
            history = build_history_buffer(time_steps, num_classes, notes_in_bar, style_hot)

            if train_all:
                target_history = build_history_buffer(time_steps, num_classes, notes_in_bar, style_hot)

            for beat, note in enumerate(melody):
                note_hot = one_hot(note, num_classes)
                beat_input = compute_beat(beat, notes_in_bar)
                completion_input = np.array([beat / (len(melody) - 1)])

                # Wrap around
                next_note_index = 0 if beat + 1 >= len(melody) else beat + 1
                next_note_hot = one_hot(melody[next_note_index], num_classes)

                history.append([note_hot, beat_input, completion_input, style_hot])

                # Yield the current input with target
                if train_all:
                    target_history.append([next_note_hot])
                    yield zip(*history), zip(*target_history)
                else:
                    yield zip(*history), [next_note_hot]


def main():
    loaded_data = load_melodies_thread(['data/edm', 'data/clean_midi'])
    np.save('data/data_cache.npy', loaded_data)

if __name__ == '__main__':
    main()
