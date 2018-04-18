import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, model):
        self.model = model

    def generate(self, seq_len=1000, show_progress=True):
        self.model.eval()
        r = trange(seq_len) if show_progress else range(seq_len)

        seq = []
        projection = nn.Linear(NUM_ACTIONS, self.model.input_size)
        # Generate first input for decoder
        x = Variable(torch.zeros(1, 1, self.model.decoder.hidden_size))
        # Generate random latent vector
        z = Variable(torch.rand(1, self.model.latent_size))

        decoder_output, decoder_hidden = self.model.decoder(x, z)
        # Append chosen event to sequence
        output_max = torch.max(F.softmax(decoder_output, dim=2), 2)
        seq.append(output_max[1].squeeze(0).data.numpy()[0])

        for _ in r:
            x = projection(decoder_output)
            z = decoder_hidden[-1]
            decoder_output, decoder_hidden = self.model.decoder(x, z)
            # Append chosen event to sequence
            output_max = torch.max(F.softmax(decoder_output, dim=2), 2)
            seq.append(output_max[1].squeeze(0).data.numpy()[0])

        return np.array(seq)

    def export(self, name='output', seq_len=1000, show_progress=True):
        """
        Export into a MIDI file.
        """
        seq = self.generate(seq_len, show_progress=show_progress)
        save_midi(name, seq)


def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--fname', default='output', help='Name of the output file')
    parser.add_argument('--model', help='Path to model file')
    parser.add_argument('--length', default=5000, type=int, help='Length of generation')
    parser.add_argument('--synth', default=False, action='store_true', help='Synthesize output in MP3')
    args = parser.parse_args()
    
    print('=== Loading Model ===')
    print('Path: {}'.format(args.model))
    settings['force_cpu'] = True
    
    model = DeepJ()

    if args.model:
        model.load_state_dict(torch.load(args.model))
    else:
        print('WARNING: No model loaded! Please specify model path.')

    print('=== Generating ===')

    fname = args.fname
    print('File: {}'.format(fname))
    generation = Generation(model)
    generation.export(name=fname, seq_len=args.length)

    if args.synth:
        data = synthesize(os.path.join(SAMPLES_DIR, fname + '.mid'))
        with open(os.path.join(SAMPLES_DIR, fname + '.mp3'), 'wb') as f:
            f.write(data)

if __name__ == '__main__':
    main()
