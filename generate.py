import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import trange

from midi_util import *
from dataset import *
from constants import *
from util import *
from model import DeepJ

def sample_timestep(model, prev_timestep, beat, states, temperature=1, batch_size=1):
    """
    Samples a single note
    """
    ## Time Axis
    note_features, states = model.time_axis(prev_timestep, beat, states)

    ## Note Axis
    # The current note being generated
    current_timestep = var(torch.zeros(batch_size, NUM_NOTES, NOTE_UNITS), volatile=True)

    for n in range(NUM_NOTES):
        output = model.note_axis(note_features, current_timestep, temperature)
        output = output.cpu().data

        # Sample note randomly
        note_on = 1 if np.random.random() <= output[0, n, 0] else 0
        current_timestep[0, n, 0] = note_on

        if note_on:
            # Sample replay
            current_timestep[0, n, 1] = 1 if np.random.random() <= output[0, n, 1] else 0
            # Volume
            current_timestep[0, n, 2] = output[0, n, 2]

    return current_timestep, states

def generate(model, name='output', num_bars=8):
    # if prime:
    #     print('Priming melody')

    model.eval()

    # Output note sequence
    note_seq = []

    # RNN state
    states = None

    # Temperature of generation
    temperature = 1
    silent_time = NOTES_PER_BAR

    # Last generated note time step
    prev_timestep = var(torch.zeros((NUM_NOTES, NOTE_UNITS)), volatile=True).unsqueeze(0)

    for t in trange(NOTES_PER_BAR * num_bars):
        beat = var(to_torch(compute_beat(t, NOTES_PER_BAR)), volatile=True).unsqueeze(0)
        current_timestep, states = sample_timestep(model, prev_timestep, beat, states, temperature=temperature)

        # Add note to note sequence
        note_seq.append(current_timestep.cpu().data[0, :].numpy())

        # if prime:
        #     prev_timestep, *_ = next(prime)
        #     prev_timestep = var(prev_timestep, volatile=True).unsqueeze(0)
        # else:
        prev_timestep = current_timestep

        # Increase temperature if silent
        if np.count_nonzero(current_timestep) == 0:
            silent_time += 1

            if silent_time >= NOTES_PER_BAR:
                temperature += 0.1
        else:
            silent_time = 0
            temperature = 1

    note_seq = np.array(note_seq)
    write_file(name, note_seq)

def write_file(name, note_seq):
    """
    Takes a list of all notes generated per track and writes it to file
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    fpath = SAMPLES_DIR + '/' + name + '.mid'
    print('Writing file', fpath)
    mf = midi_encode(unclamp_midi(note_seq))
    midi.write_midifile(fpath, mf)

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('path', help='Path to model file')
    parser.add_argument('--bars', default=16, type=int, help='Bars of generation')
    parser.add_argument('--debug', default=False, action='store_true', help='Use training data as input')
    args = parser.parse_args()

    # primer = None
    # if args.debug:
    #     print('=== Loading Data ===')
    #     primer = data_it(process(load_styles()))

    print('=== Loading Model ===')
    print('Path: {}'.format(args.path))
    print('GPU: {}'.format(torch.cuda.is_available()))
    model = DeepJ()
    
    if torch.cuda.is_available():
        model.cuda()

    print('WARNING: No model loaded!')
    # model.load_state_dict(torch.load(args.path))

    print('=== Generating ===')
    generate(model, num_bars=args.bars)

if __name__ == '__main__':
    main()
