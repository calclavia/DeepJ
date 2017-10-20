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
from midi_io import load_midi
from util import *

def load(styles=STYLES):
    """
    Loads all music styles into a list of compositions
    """
    style_seqs = []
    for style in styles:
        # Parallel process all files into a list of music sequences
        style_seq = []
        seq_len_sum = 0

        for f in tqdm(get_all_files([style])):
            try:
                seq = load_midi(f)
                style_seq.append(to_torch(seq))
                seq_len_sum += len(seq)
            except Exception as e:
                print('Unable to load {}'.format(f))
        
        style_seqs.append(style_seq)
        print('Loading {} MIDI file(s) with average event count {}'.format(len(style_seq), seq_len_sum / len(style_seq)))
    return style_seqs

def process(style_seqs, seq_len=SEQ_LEN):
    """
    Process data. Takes a list of styles and flattens the data, returning the necessary tags.
    """
    # Flatten into compositions list
    seqs = [s for y in style_seqs for s in y]
    style_tags = torch.stack([to_torch(pad_before(one_hot(s, NUM_STYLES))) for s, y in enumerate(style_seqs) for x in y])
    return seqs, style_tags

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
    seqs, *_ = data

    # List of composition and their sequence start indices
    it_list = []

    for c, seq in enumerate(seqs):
        for t in range(0, len(seq) - 1 - seq_len, SEQ_SPLIT):
            it_list.append((c, t))

    return it_list

def sampler(data, it_list, seq_len=SEQ_LEN):
    """
    Generates sequences of data.
    """
    seqs, style_tags = data

    if len(seqs) == 0:
        raise 'Insufficient training data.'

    # A list of iteration indices that specify the iteration order
    it_shuffled = random.sample(it_list, len(it_list))

    for c, t in it_shuffled:
        yield (seqs[c][t:t+seq_len], style_tags[c])

def batcher(sampler, batch_size=BATCH_SIZE):
    """
    Bundles samples into batches
    """
    batch = []

    for sample in sampler:
        batch.append(sample)
        
        if len(batch) == batch_size:
            # Convert batch
            yield [torch.stack(x) for x in zip(*batch)]
            batch = []

    # Yield the remaining batch!
    yield [torch.stack(x) for x in zip(*batch)]

def data_it(data, seq_len=SEQ_LEN):
    """
    Iterates through each note in all songs.
    """
    seqs, style_tags = data

    for c in range(len(seqs)):
        note_seq = seqs[c]
        style_tag = style_tags[c]

        for t in range(0, len(note_seq)):
            yield (note_seq[t], style_tag)