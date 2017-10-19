"""
Preprocesses MIDI files
"""
import math
import numpy as np
import torch

import numpy
import math
import random
from tqdm import tqdm
import multiprocessing

from constants import *
from midi_util import load_midi
from util import *

def compute_beat(beat, notes_in_bar):
    return one_hot(beat % notes_in_bar, notes_in_bar)

def compute_completion(beat, len_melody):
    return torch.tensor([beat / len_melody])

def load_styles(styles=STYLES):
    """
    Loads all music styles into a list of compositions
    """
    style_seqs = []
    for style in tqdm(styles):
        # Parallel process all files into a list of music sequences
        style_seq = [load_midi(f) for f in tqdm(get_all_files([style]))]
        style_seqs.append(style_seq)

    return style_seqs

def extract_beat(compositions):
    """
    Given a list of music compositions, tag each time step with beat data.
    """
    beat_tags = []

    for comp in compositions:
        beats = np.array([compute_beat(t, NOTES_PER_BAR) for t in range(len(comp))])
        beat_tags.append(to_torch(beats))
    return beat_tags

def process(style_seqs, seq_len=SEQ_LEN):
    """
    Process data. Takes a list of styles and flattens the data, returning the necessary tags.
    """
    # Flatten into compositions list
    seqs = [s for y in style_seqs for s in y]
    style_tags = torch.stack([to_torch(one_hot(s, NUM_STYLES)) for s, y in enumerate(style_seqs) for x in y])

    seqs = [to_torch(pad_before(clamp_midi(x))) for x in seqs if len(x) > seq_len]
    beat_tags = extract_beat(seqs)
    return seqs, beat_tags, style_tags

def validation_split(it_list, split=0.1):
    """
    Splits the data iteration list into training and validation indices
    """
    # Shuffle data
    random.shuffle(it_list)

    num_val = int(math.ceil(len(it_list) * split))
    training_indicies = it_list[:-num_val]
    validation_indicies = it_list[-num_val:]

    assert len(validation_indicies) == num_val
    assert len(training_indicies) == len(it_list) - num_val
    return training_indicies, validation_indicies

def iteration_indices(data, seq_len=SEQ_LEN):
    """
    Returns a list of tuple, which are the iteration indices.
    """
    note_seqs, *_ = data

    # List of composition and their sequence start indices
    it_list = []

    for c, seq in enumerate(note_seqs):
        for t in range(0, len(seq) - 1 - seq_len, NOTES_PER_BAR):
            it_list.append((c, t))

    return it_list

def sampler(data, it_list, seq_len=SEQ_LEN):
    """
    Generates sequences of data.
    """
    note_seqs, beat_tags, style_tags = data

    if len(note_seqs) == 0:
        raise 'Insufficient training data.'

    # A list of iteration indices that specify the iteration order
    it_shuffled = random.sample(it_list, len(it_list))

    for c, t in it_shuffled:
        yield (note_seqs[c][t:t+seq_len], \
               beat_tags[c][t:t+seq_len], \
               style_tags[c])

def data_it(data, seq_len=SEQ_LEN):
    """
    Iterates through each note in all songs.
    """
    note_seqs, beat_tags, style_tags = data

    for c in range(len(note_seqs)):
        note_seq = note_seqs[c]
        beat_seq = beat_tags[c]
        style_tag = style_tags[c]

        for t in range(0, len(note_seq)):
            yield (note_seq[t], beat_seq[t], style_tag)

def batcher(sampler, batch_size=BATCH_SIZE):
    """
    Bundles samples into batches
    """
    batch = []

    for sample in sampler:
        batch.append(sample)

        if len(batch) == batch_size:
            # Convert batch
            yield [var(torch.stack(x)) for x in zip(*batch)]
            batch = []

    # Yield the remaining batch!
    yield [var(torch.stack(x)) for x in zip(*batch)]

def clamp_midi(sequence):
    """
    Clamps the midi base on the MIN and MAX notes
    """
    return sequence[:, MIN_NOTE:MAX_NOTE, :]

def unclamp_midi(sequence):
    """
    Restore clamped MIDI sequence back to MIDI note values
    """
    return np.pad(sequence, ((0, 0), (MIN_NOTE, 0), (0, 0)), 'constant')
