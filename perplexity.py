import numpy as np
import argparse
import heapq

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

def main():
    parser = argparse.ArgumentParser(description='Evaluates model perplexity.')
    parser.add_argument('--path', help='Path to model file')
    args = parser.parse_args()

    print('=== Loading Model ===')
    print('Path: {}'.format(args.path))
    print('GPU: {}'.format(torch.cuda.is_available()))
    
    model = DeepJ().cuda()

    if args.path:
        model.load_state_dict(torch.load(args.path))
    else:
        print('WARNING: No model loaded! Please specify model path.')

    print('Loading data...')
    data = process(load())

    print('=== Perplexity ===')
    print('Sequences: {}'.format(len(data[0])))
    
    sum_logprob = 0
    num_tokens = 0

    t = tqdm(zip(*data), total=len(data[0]))
    for seq, style in t:
        # Trim sequence to be power of 2
        trim_len = 2 ** int(np.log2(seq.size(0))) + 1
        seq = seq[:trim_len]
        
        # Feed entire sequence in and get the log probability of the correct tokens
        styles = var(to_torch(one_hot(style, NUM_STYLES)), volatile=True).unsqueeze(0)

        # One hot encoding buffer that you create out of the loop and just keep reusing
        seq_onehot = torch.FloatTensor(seq.size(0) - 1, NUM_ACTIONS)
        seq_onehot.zero_()
        seq_onehot.scatter_(1, seq[:-1].unsqueeze(1), 1)

        inputs = var(seq_onehot, volatile=True).unsqueeze(0)
        output, _ = model(inputs, styles, None)

        # Remove batch dim
        output = F.log_softmax(output.squeeze(0), dim=1)
        target = Variable(seq[1:]).cuda().unsqueeze(1)
        probs = output.gather(1, target)

        # Compute statistics
        sum_logprob += probs.sum().cpu().data[0]
        num_tokens += seq.size(0)
        entropy = -(1.0 / num_tokens) * (sum_logprob)
        perplexity = pow(2.0, entropy)
        t.set_postfix(entropy=entropy, perplexity=perplexity)

    print('Perplexity: {}'.format(perplexity))

if __name__ == '__main__':
    main()