import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import *
from constants import *

def train(model, data_generator):
    """
    Trains a model on multiple seq batches by iterating through a generator.
    """
    step = 1
    # Number of training steps per epoch
    epoch_len = 1000
    # Keep tracks of all losses in each epoch
    all_losses = []
    total_loss = 0

    t = tqdm(data_generator, total=epoch_len)

    for data in t:
        loss = train_step(model, *data)
        total_loss += loss
        avg_loss = total_loss / step
        t.set_postfix(loss=avg_loss)

        if step % epoch_len == 0:
            all_losses.append(avg_loss)
            total_loss = 0

            # Draw graph
            plt.clf()
            plt.plot(all_losses)
            plt.savefig('out/loss.png')

            # Save model
            torch.save(model, 'out/checkpoint.pt')

            step = 0

        step += 1

def train_step(model, note_seq):
    """
    Trains the model on a single batch of sequence.
    """
    # TODO: Verify loss correctness
    criterion = nn.NLLLoss()
    # TODO: Clip gradient if needed.
    optimizer = optim.Adam(model.parameters())

    # Initialize hidden states
    states = model.init_states(BATCH_SIZE)

    # Zero out the gradient
    model.zero_grad()

    loss = 0

    # Iterate through the entire sequence
    for i in range(note_seq.size()[0] - 1):
        # TODO: We can apply custom input based on mistakes.
        output, states = model(note_seq[i], states)
        loss += criterion(output, sequence[i + 1])

    loss.backward()
    optimizer.step()
    return loss.data[0] / input_line_tensor.size()[0]

def main():
    print('=== Dataset ===')
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Loading...')
    style_seqs = load_styles()
    print('Creating data generator...')
    generator = batcher(sampler(style_seqs))
    print('=== Training ===')
    train()

if __name__ == '__main__':
    main()
