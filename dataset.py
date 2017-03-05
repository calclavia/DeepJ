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
import random
from constants import styles
from midi_util import *

def process_melody(melody):
    """
    Converts a melody data to be compatible with the neural network.
    """
    res = []
    for x in melody:
        if x >= MAX_NOTE:
            res.append(NO_EVENT)
        elif x >= 0:
            res.append(x - MIN_NOTE + MIN_CLASS)
        else:
            # Apply shift to NOTE_OFF and NO_EVENT
            res.append(abs(x) - 1)
    return res


def load_melody(fname, transpose=None):
    try:
        seq_pb = midi_io.midi_to_sequence_proto(fname)
        melody = melodies_lib.midi_file_to_melody(seq_pb, steps_per_quarter=NOTES_PER_BEAT)
        melody.squash(MIN_NOTE, MAX_NOTE, transpose)
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
                if os.path.isfile(fname) and fname.endswith('.mid'):
                    potential_files.append(fname)
    return potential_files


def load_process(f, doProcess, transpose):
    melody = load_melody(f, transpose)
    if melody:
        return f, (process_melody(melody) if doProcess else melody)
    else:
        return f, None


def load_melodies(paths, transpose=None, process=True, limit=None, shuffle=True, named=False):
    assert len(paths) > 0
    files = get_all_files(paths)

    if shuffle:
        random.shuffle(files)

    if limit is not None:
        print('Limiting to {} files'.format(limit))
        files = files[:limit]

    print('Loading melodies from {} files'.format(len(files)))
    res = Parallel(n_jobs=8, verbose=5, backend='threading')(delayed(load_process)(f, process, transpose) for f in files)

    out = []
    skipped = 0
    for result in res:
        if result[1] == None or len(result[1]) <= 1:
            skipped += 1
        else:
            if named:
                out.append(result)
            else:
                out.append(result[1])

    print('Loaded {} melodies (skipped {})'.format(len(out), skipped))
    return out


def compute_beat(beat, notes_in_bar):
    # TODO: Compare methods
    # Angle method
    # angle = (beat % notes_in_bar) / notes_in_bar * 2 * math.pi
    # return np.array([math.cos(angle), math.sin(angle)])
    return one_hot(beat % notes_in_bar, notes_in_bar)

def compute_completion(beat, len_melody):
    return np.array([beat / (len_melody - 1)])

def build_history_buffer(time_steps, num_classes, notes_in_bar, style_hot, prime_beats=True):
    history = deque(maxlen=time_steps)
    current_beat = NOTES_PER_BAR - 1
    for i in range(time_steps):
        assert current_beat >= 0
        beat_input = compute_beat(current_beat, notes_in_bar)
        history.appendleft([
            np.zeros(num_classes),
            beat_input if prime_beats else np.zeros_like(beat_input),
            np.zeros(1),
            style_hot
        ])

        current_beat -= 1
        if current_beat < 0:
            current_beat = NOTES_PER_BAR - 1
    return history

def opt_one_hot(x, num_classes):
    """
    Creates a one hot representation if needed.
    """
    return one_hot(x, num_classes) if isinstance(x, int) else x

def data_gen(music,
            style_hot,
            time_steps,
            num_classes,
            notes_in_bar,
            target_all):
    """
    Generates training and label data for a given music sequence.

    Return:
        A list of samples. Each sample consist of inputs for each time step
        and a target level as a tuple.
        samples:
            inputs: (input, time_steps, ?)
            target: (?)
    """
    # Recurrent history
    history = build_history_buffer(time_steps, num_classes, notes_in_bar, style_hot)

    if target_all:
        target_history = build_history_buffer(time_steps, num_classes, notes_in_bar, style_hot)

    music = list(music)

    for beat, note in enumerate(music):
        beat_input = compute_beat(beat, notes_in_bar)
        completion_input = compute_completion(beat, len(music))
        
        predict_note = opt_one_hot(note, num_classes)

        # Yield the current input with target
        if target_all:
            # Yield a list of targets for each time step
            target_history.append([predict_note])
            yield [list(x) for x in zip(*history)], [list(x) for x in zip(*target_history)]
        else:
            # Yield the target to predict
            yield [list(x) for x in zip(*history)], [predict_note]

        history.append(
            [predict_note, beat_input, completion_input, style_hot]
        )

def stateless_gen(music_styles,
                    time_steps,
                    num_classes,
                    notes_in_bar,
                    target_all=False):
    """
    Generates data for a stateless RNN.
    Return:
        A list of data samples for all styles
    """
    for s, style in enumerate(music_styles):
        style_hot = one_hot(s, len(music_styles))

        for melody in style:
            for x in data_gen(melody, style_hot, time_steps, num_classes, notes_in_bar, target_all):
                yield x

def stateful_gen(music_styles, time_steps, batch_size, num_classes=NUM_CLASSES, notes_in_bar=NOTES_PER_BAR, target_all=False):
    """
    For every single melody style, yield the melody along
    with its contextual inputs.

    Return:
        A list of music sequences
    """
    # Process the data into a list of sequences.
    # Each sequence contains input tracks
    # Each training sequence is a tuple of various inputs, including contexts
    for s, style in enumerate(music_styles):
        style_hot = one_hot(s, len(music_styles))

        for melody in style:
            m_data = list(data_gen(melody, style_hot, time_steps, num_classes, notes_in_bar, target_all))
            # A list of sample inputs and targets
            samples, targets = zip(*m_data)

            # (samples, inputs, timesteps, ?) -> (batches, batch_size, inputs, timesteps, ?)
            # Batchify input
            batches = []
            current_batch = None

            for i, sample in enumerate(samples):
                if i % batch_size == 0:
                    if current_batch is not None:
                        batches.append(current_batch)
                    current_batch = []

                current_batch.append(sample)

            # (batches, batch_size, inputs, timesteps, ?) -> (batches, inputs, batch_size, timesteps, ?)
            batches = [[np.array(list(batch_input)) for batch_input in zip(*batch)] for batch in batches]
            targets = np.squeeze(targets)
            truncated = (len(targets) // batch_size) * batch_size
            targets = np.swapaxes(np.split(targets[:truncated], batch_size), 0, 1)
            yield batches, targets

def load_styles(transpose=None, limit=None):
    # A list of styles, each containing melodies
    return [load_melodies([style], limit=limit, transpose=transpose) for style in styles]

def load_music_styles():
    """
    Loads style for a full music score (not just melody)
    """
    music_styles = []
    for s_file in styles:
        print('Searching {}'.format(s_file))
        files = get_all_files([s_file])
        print('Loading music from {} files'.format(len(files)))
        res = Parallel(n_jobs=8, verbose=5, backend='threading')(delayed(load_midi)(f) for f in files)

        # Trim the representation to be between MIN and MAX note
        # TODO: There's 2 extra to fill in compatibility with melody rnn
        # TODO: Volume input
        res = [np.ceil(m[:, MIN_NOTE:MAX_NOTE+2]) for m in res]
        music_styles.append(res)

    return music_styles

def process_stateful(music_styles, time_steps, shuffle=True, batch_size=1):
    print('Processing dataset')
    return list(stateful_gen(music_styles, time_steps, batch_size=batch_size))

def process_stateless(music_styles, time_steps):
    print('Processing dataset')
    input_set, target_set = zip(*list(stateless_gen(music_styles, time_steps, NUM_CLASSES, NOTES_PER_BAR)))
    input_set = [np.array(i) for i in zip(*input_set)]
    target_set = [np.array(i) for i in zip(*target_set)]
    return input_set, target_set
