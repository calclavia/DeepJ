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
    
def generate(model, start_token=torch.tensor([0]), max_len=const.SEQ_LEN - 1, temperature=1, auto_break=False):
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
        
        if auto_break and sampled_token == const.TOKEN_EOS:
            break

    return zip(*outputs)

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
            seq, *_ = generate(model, max_len=args.length, temperature=args.temperature, auto_break=True)
            seq = [x.item() for x in seq]
            midi_file = tokens_to_midi(seq)
            midi_file.save('out/samples/' + fname + '.mid')

            if args.synth:
                data = synthesize(os.path.join(SAMPLES_DIR, fname + '.mid'))
                with open(os.path.join(SAMPLES_DIR, fname + '.mp3'), 'wb') as f:
                    f.write(data)

if __name__ == '__main__':
    main()
