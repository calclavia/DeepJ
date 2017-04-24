"""
Preprocesses MIDI files
"""
import numpy as np
import torch
from torch.autograd import Variable

import numpy
import math
import random
from tqdm import tqdm
import multiprocessing

from constants import *
from midi_util import load_midi
from util import get_all_files, one_hot

def compute_beat(beat, notes_in_bar):
    # TODO: Compare methods
    # Angle method
    # angle = (beat % notes_in_bar) / notes_in_bar * 2 * math.pi
    # return np.array([math.cos(angle), math.sin(angle)])
    return one_hot(beat % notes_in_bar, notes_in_bar)

def compute_completion(beat, len_melody):
    return torch.tensor([beat / len_melody])

def random_subseq(sequence, length, division_len=NOTES_PER_BAR):
    """
    Returns the range of a random subseq
    """
    # Make random starting position of sequence
    # Will never pick the last element in sequence
    end = len(sequence) - 1 - length

    if end == 0:
        # No choice
        return (0, length)

    start = random.randrange(0, end, division_len)
    return (start, start + length)

def random_comp_subseq(compositions, length, division_len=NOTES_PER_BAR):
    """
    Returns a random music subsequence from a list of compositions
    """
    comp_index = random.randint(0, len(compositions) - 1)
    subseq_range = random_subseq(compositions[comp_index], length, division_len)
    return comp_index, subseq_range

def load_styles(styles=STYLES):
    """
    Loads all music styles into a list of compositions
    """
    style_seqs = []
    for style in tqdm(styles):
        # Parallel process all files into a list of music sequences
        style_seq = [list(load_midi(f)) for f in tqdm(get_all_files([style]))]
        style_seqs.append(style_seq)
    return style_seqs

def extract_beat(compositions):
    """
    Given a list of music compositions, tag each time step with beat data.
    """
    beat_tags = []

    for comp in compositions:
        beat_tags.append(np.array([compute_beat(t, NOTES_PER_BAR) for t in range(len(comp))]))
    return beat_tags

def sampler(style_seqs, seq_len=SEQ_LEN):
    """
    Generates training samples.
    """
    # Flatten into compositions list
    flat_seq = [x for y in style_seqs for x in y]
    style_tags = torch.stack([torch.from_numpy(one_hot(s, NUM_STYLES)) for s, y in enumerate(style_seqs) for x in y])

    note_seqs, replay_seqs = zip(*flat_seq)

    note_seqs = [clamp_midi(x) for x in note_seqs if len(x) > seq_len]
    replay_seqs = [clamp_midi(x) for x in replay_seqs if len(x) > seq_len]
    beat_tags = extract_beat(note_seqs)

    if len(note_seqs) == 0:
        raise 'Insufficient training data.'

    while True:
        comp_index, r = random_comp_subseq(note_seqs, seq_len, 1)

        yield (torch.from_numpy(note_seqs[comp_index][r[0]:r[1]]).float(), \
               torch.from_numpy(replay_seqs[comp_index][r[0]:r[1]]).float(), \
               torch.from_numpy(beat_tags[comp_index][r[0]:r[1]]).float(), \
               style_tags[comp_index].float())

def batcher(sampler, batch_size=BATCH_SIZE):
    """
    Bundles samples into batches
    """
    batch = []

    for sample in sampler:
        batch.append(sample)

        if len(batch) == batch_size:
            # Convert batch
            yield [Variable(torch.stack(x)) for x in zip(*batch)]
            batch = []

def clamp_midi(sequence):
    """
    Clamps the midi base on the MIN and MAX notes
    """
    sequence = np.minimum(np.ceil(sequence[:, MIN_NOTE:MAX_NOTE]), 1)
    assert (sequence >= 0).all()
    assert (sequence <= 1).all()
    return sequence

def unclamp_midi(sequence):
    """
    Restore clamped MIDI sequence back to MIDI note values
    """
    return np.concatenate((np.zeros((len(sequence), MIN_NOTE)), sequence), axis=1)
