import numpy as np
import argparse
import heapq

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import trange

from midi_io import *
from dataset import *
from constants import *
from util import *
from model import DeepJ
    
def generate(model, start_token=torch.tensor([0]), max_len=const.SEQ_LEN - 1, temperature=1):
    """
    Generates samples up to max length
    """
    outputs = []
    memory = None

    for _ in range(max_len):
        if len(outputs) == 0:
            prev_token = start_token
        else:
            prev_token = outputs[-1][0]
        
        logits, memory, hidden = model(prev_token, memory)
        probs = torch.softmax(logits / temperature, dim=-1)

        # Sample action
        sampled_token = probs.multinomial(1).detach().squeeze(-1)
        
        outputs.append((sampled_token, hidden))
        
        # TODO: Re-enable this?
        # if sampled_token == const.TOKEN_EOS:
        #     break
    return zip(*outputs)
    
class Generation():
    """
    Represents a music generation sequence
    """

    def __init__(self, model, style=None, default_temp=1):
        self.model = model
        self.style = style

        # Temperature of generation
        self.default_temp = default_temp
        self.temperature = self.default_temp

        # Model parametrs
        self.outputs = [const.TOKEN_EOS]
        self.state = None

    def step(self):
        """
        Generates the next set of beams
        """
        # Iterate through the beam
        prev_event = torch.tensor(self.outputs[-1])
        
        prev_event = prev_event.unsqueeze(0)
        logits, self.state = self.model(prev_event, self.state)
        probs = torch.softmax(logits / self.temperature, dim=-1)

        # Sample action
        output = probs.multinomial(1)
        event = output.item()
        
        self.outputs.append(event)

    def generate(self, seq_len, show_progress=True):
        self.model.eval()
        r = trange(seq_len) if show_progress else range(seq_len)

        for _ in r:
            self.step()

            if self.outputs[-1] == const.TOKEN_EOS:
                break

        return np.array(self.outputs)

def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('model', help='Path to model file')
    parser.add_argument('--fname', default='output', help='Name of the output file')
    parser.add_argument('--length', default=5000, type=int, help='Length of generation')
    parser.add_argument('--style', default=None, type=int, nargs='+', help='Specify the styles to mix together. By default will generate all possible styles.')
    parser.add_argument('--temperature', default=1, type=float, help='Temperature of generation')
    parser.add_argument('--synth', default=False, action='store_true', help='Synthesize output in MP3')
    args = parser.parse_args()

    styles = []

    if args.style:
        # Custom style
        styles = [np.mean([one_hot(i, NUM_STYLES) for i in args.style], axis=0)]
    else:
        # Generate all possible style
        styles = [one_hot(i, NUM_STYLES) for i in range(len(STYLES))]
    
    print('=== Loading Model ===')
    print('Path: {}'.format(args.model))
    print('Temperature: {}'.format(args.temperature))
    print('Styles: {}'.format(styles))
    settings['force_cpu'] = True
    
    model = DeepJ()

    if args.model:
        model.load_state_dict(torch.load(args.model))
    else:
        print('WARNING: No model loaded! Please specify model path.')

    print('=== Generating ===')

    with torch.no_grad():
        for style in styles:
            fname = args.fname + str(list(style))
            print('File: {}'.format(fname))
            generation = Generation(model, style=style, default_temp=args.temperature)
            seq = generation.generate(seq_len=args.length)
            print(seq)
            midi_file = tokens_to_midi(seq.tolist())
            midi_file.save('out/samples/' + fname + '.mid')

            if args.synth:
                data = synthesize(os.path.join(SAMPLES_DIR, fname + '.mid'))
                with open(os.path.join(SAMPLES_DIR, fname + '.mp3'), 'wb') as f:
                    f.write(data)

if __name__ == '__main__':
    main()
