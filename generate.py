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

class Generation():
    """
    Represents a music generation sequence
    """

    def __init__(self, model, style=None, primer=None, default_temp=1):
        self.model = model

        # Pick a random style
        self.style = style if style is not None else one_hot(np.random.randint(0, NUM_STYLES), NUM_STYLES)

        # Temperature of generation
        self.temperature = default_temp
        # How much time of silence
        self.silent_time = SILENT_LENGTH

        # Model parametrs
        self.prev_out = var(torch.zeros((1, NUM_ACTIONS)), volatile=True)
        self.states = None

    def next(self):
        """
        Generates the next event
        """
        # Create variables
        style = var(to_torch(self.style), volatile=True)
        output, self.states = self.model.generate(self.prev_out, style, self.states, temperature=self.temperature)
        # Add note to note sequence
        self.prev_out = var(to_torch(output), volatile=True)

        # Increase temperature if silent
        """
        if np.count_nonzero(current_timestep) == 0:
            self.silent_time += 1

            if self.silent_time >= SILENT_LENGTH:
                self.temperature += 0.1
        else:
            self.silent_time = 0
            self.temperature = default_temp
        """
        return output[0]

    def export(self, name='output', seq_len=200):
        """
        Export into a MIDI file.
        """
        seq = np.array([self.next() for t in trange(seq_len)])
        save_midi(name, seq)

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--path', help='Path to model file')
    parser.add_argument('--length', default=16, type=int, help='Length of generation')
    parser.add_argument('--style', default=None, type=int, nargs='+', help='Styles to mix together')
    parser.add_argument('--debug', default=False, action='store_true', help='Use training data as input')
    args = parser.parse_args()

    primer = None
    if args.debug:
        print('=== Loading Data ===')
        primer = data_it(process(load()))

    style = None

    if args.style:
        # Custom style
        style = [np.mean([one_hot(i, NUM_STYLES) for i in args.style], axis=0)]

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
    Generation(model, style=style, primer=primer).export(seq_len=args.length)

if __name__ == '__main__':
    main()
