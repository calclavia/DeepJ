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

def generate(model, name='output', num_bars=4, primer=None, default_temp=0.9):
    model.eval()

    # Output note sequence
    note_seq = []

    # RNN state
    states = None

    # Temperature of generation
    temperature = default_temp
    silent_time = NOTES_PER_BAR

    if primer:
        print('Priming melody')
        prev_timestep, *_ = next(primer)
        prev_timestep = var(prev_timestep, volatile=True).unsqueeze(0)
    else:
        # Last generated note time step
        prev_timestep = var(torch.zeros((NUM_NOTES, NOTE_UNITS)), volatile=True).unsqueeze(0)

    for t in trange(NOTES_PER_BAR * num_bars):
        beat = var(to_torch(compute_beat(t, NOTES_PER_BAR)), volatile=True).unsqueeze(0)
        current_timestep, states = model.generate(prev_timestep, beat, states, temperature=temperature)

        cur_timestep_numpy = current_timestep.cpu().data.numpy()
        # Add note to note sequence
        note_seq.append(cur_timestep_numpy[0, :])

        if primer:
            # Inject training data to input
            prev_timestep, *_ = next(primer)
            prev_timestep = var(prev_timestep, volatile=True).unsqueeze(0)
        else:
            prev_timestep = current_timestep

        # Increase temperature if silent
        if np.count_nonzero(cur_timestep_numpy) == 0:
            silent_time += 1

            if silent_time >= NOTES_PER_BAR:
                temperature += 0.1
        else:
            silent_time = 0
            temperature = default_temp

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
    parser.add_argument('--path', help='Path to model file')
    parser.add_argument('--bars', default=16, type=int, help='Bars of generation')
    parser.add_argument('--debug', default=False, action='store_true', help='Use training data as input')
    args = parser.parse_args()

    primer = None
    if args.debug:
        print('=== Loading Data ===')
        primer = data_it(process(load_styles()))

    print('=== Loading Model ===')
    print('Path: {}'.format(args.path))
    print('GPU: {}'.format(torch.cuda.is_available()))
    settings['force_cpu'] = True
    
    model = DeepJ()

    if args.path:
        model.load_state_dict(torch.load(args.path))
    else:
        print('WARNING: No model loaded! Please specify model path.')

    print('=== Generating ===')
    generate(model, num_bars=args.bars, primer=primer)

if __name__ == '__main__':
    main()
