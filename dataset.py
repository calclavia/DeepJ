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

def random_subseq(sequence, length, division_len):
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

def random_comp_subseq(compositions, length, division_len):
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
        beats = np.array([compute_beat(t, NOTES_PER_BAR) for t in range(len(comp))])
        beat_tags.append(torch.from_numpy(beats).float())
    return beat_tags

def process(style_seqs, seq_len=SEQ_LEN):
    """
    Process data
    """
    # Flatten into compositions list
    flat_seq = [x for y in style_seqs for x in y]
    style_tags = torch.stack([torch.from_numpy(one_hot(s, NUM_STYLES)) for s, y in enumerate(style_seqs) for x in y]).float()

    note_seqs, replay_seqs = zip(*flat_seq)

    note_seqs = [torch.from_numpy(clamp_midi(x)).float() for x in note_seqs if len(x) > seq_len]
    replay_seqs = [torch.from_numpy(clamp_midi(x)).float() for x in replay_seqs if len(x) > seq_len]
    beat_tags = extract_beat(note_seqs)
    return note_seqs, replay_seqs, beat_tags, style_tags

def sampler(data, seq_len=SEQ_LEN):
    """
    Generates training samples.
    """
    note_seqs, replay_seqs, beat_tags, style_tags = data

    if len(note_seqs) == 0:
        raise 'Insufficient training data.'

    while True:
        comp_index, r = random_comp_subseq(note_seqs, seq_len, 1)

        yield (note_seqs[comp_index][r[0]:r[1]], \
               replay_seqs[comp_index][r[0]:r[1]], \
               beat_tags[comp_index][r[0]:r[1]], \
               style_tags[comp_index])

def data_it(data, seq_len=SEQ_LEN):
    """
    Iterates through each note in all songs.
    """
    note_seqs, replay_seqs, beat_tags, style_tags = data

    for c in range(len(note_seqs)):
        note_seq = note_seqs[c]
        replay_seq = replay_seqs[c]
        beat_seq = beat_tags[c]
        style_tag = style_tags[c]

        for t in range(0, len(note_seq)):
            yield (note_seq[t], replay_seq[t], beat_seq[t], style_tag)

def batcher(sampler, batch_size=BATCH_SIZE):
    """
    Bundles samples into batches
    """
    batch = []

    for sample in sampler:
        batch.append(sample)

        if len(batch) == batch_size:
            # Convert batch
            yield [Variable(torch.stack(x)).cuda() for x in zip(*batch)]
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
