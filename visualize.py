import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import os
from model import DeepJ
from tqdm import tqdm
import numpy as np
import dataset
import constants as const
import argparse

def main():
    parser = argparse.ArgumentParser(description='Visualize model latent space.')
    parser.add_argument('model', help='Path to model file')
    parser.add_argument('--batch_size', default=8)
    args = parser.parse_args()

    data, labels = dataset.process(dataset.load())
    labels = labels.cpu().numpy().tolist()

    model = DeepJ().cuda().half()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    num_data = len(data) - (len(data) % args.batch_size)

    with torch.no_grad():
        with open(os.path.join(const.OUT_DIR, 'latent.tsv'), 'w') as f:
            for i in tqdm(range(0, num_data, args.batch_size)):
                # Create batch of sequences
                d = data[i:i + args.batch_size]

                # Sort input so it can be padded.
                lengths = [len(x) for x in d]
                ordering = list(np.argsort(lengths))[::-1]
                sorted_d = [d[i] for i in ordering]
                d = pad_sequence(sorted_d, batch_first=True).cuda()
                # d = d[:, :const.SEQ_LEN]

                d = model.embd(d)
                means, _, _ = model.encoder(d, None)
                means = means.cpu().numpy().tolist()

                # Unsort output based on the sorted ordering.
                means = [means[ordering.index(i)] for i in range(args.batch_size)]

                for m in means:
                    f.write('\t'.join(map(str, m)) + '\n')

    with open(os.path.join(const.OUT_DIR, 'labels.tsv'), 'w') as f:
        f.write('\n'.join(map(str, labels[:num_data])))

if __name__ == '__main__':
    main()
