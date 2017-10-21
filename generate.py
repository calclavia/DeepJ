import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import trange

from midi_io import *
from dataset import *
from constants import *
from util import *
from model import DeepJ

def generate(model, name='output', seq_len=200, primer=None, default_temp=1):
    model.eval()

    # Output note sequence
    seq = []

    # RNN state
    states = None

    # Temperature of generation
    temperature = default_temp
    silent_time = SILENT_LENGTH

    if primer:
        print('Priming melody')
        prev_timestep, *_ = next(primer)
        prev_timestep = var(prev_timestep, volatile=True).unsqueeze(0)
    else:
        # Last generated note time step
        prev_timestep = var(torch.zeros((1, NUM_ACTIONS)), volatile=True)

    for t in trange(seq_len):
        current_timestep, states = model.generate(prev_timestep, states, temperature=temperature)

        # Add note to note sequence
        seq.append(current_timestep[0])

        if primer:
            # Inject training data to input
            prev_timestep, *_ = next(primer)
            prev_timestep = var(prev_timestep, volatile=True).unsqueeze(0)
        else:
            prev_timestep = var(to_torch(current_timestep), volatile=True)

        # Increase temperature if silent
        """
        if np.count_nonzero(current_timestep) == 0:
            silent_time += 1

            if silent_time >= SILENT_LENGTH:
                temperature += 0.1
        else:
            silent_time = 0
            temperature = default_temp
        """

    seq = np.array(seq)
    save_midi(name, seq)

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--path', help='Path to model file')
    parser.add_argument('--length', default=16, type=int, help='Length of generation')
    parser.add_argument('--debug', default=False, action='store_true', help='Use training data as input')
    args = parser.parse_args()

    primer = None
    if args.debug:
        print('=== Loading Data ===')
        primer = data_it(process(load()))

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
    generate(model, seq_len=args.length, primer=primer)

if __name__ == '__main__':
    main()
