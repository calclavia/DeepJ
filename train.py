import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import *
from constants import *
from model import DeepJ
from generate import generate

def train(model, data_generator):
    """
    Trains a model on multiple seq batches by iterating through a generator.
    """
    model.train()

    step = 1
    # Number of training steps per epoch
    epoch = 1
    epoch_len = 500
    # Keep tracks of all losses in each epoch
    all_losses = []
    total_loss = 0

    t = tqdm(total=epoch_len)

    for data in data_generator:
        t.set_description('Epoch {}'.format(epoch))

        loss = train_step(model, *data)

        total_loss += loss
        avg_loss = total_loss / step
        t.set_postfix(loss=avg_loss)
        t.update()

        if step % epoch_len == 0:
            all_losses.append(avg_loss)
            total_loss = 0

            # Draw graph
            plt.clf()
            plt.plot(all_losses)
            plt.savefig(OUT_DIR + '/loss.png')

            # Save model
            torch.save(model, OUT_DIR + '/model.pt')

            # Generate
            generate(model, name='epoch_' + str(epoch))

            step = 0
            epoch += 1

            t.close()
            t = tqdm(total=epoch_len)

        step += 1

def train_step(model, note_seq, replay_seq, beat_seq, style):
    """
    Trains the model on a single batch of sequence.
    """
    criterion = nn.BCELoss()
    # TODO: Clip gradient if needed.
    optimizer = optim.Adam(model.parameters())

    # Zero out the gradient
    optimizer.zero_grad()

    loss = 0
    seq_len = note_seq.size()[1]

    # Initialize hidden states
    states = None

    # Iterate through the entire sequence
    for i in range(seq_len - 1):
        # TODO: We can apply custom input based on mistakes.
        targets = note_seq[:, i + 1]
        output, states = model(note_seq[:, i], beat_seq[:, i], states, targets)
        loss += criterion(output, targets)

    loss.backward()
    optimizer.step()
    return loss.data[0] / seq_len

def main():
    print('=== Dataset ===')
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Loading...')
    generator = batcher(sampler(load_styles()))
    print()
    print('=== Training ===')
    print('GPU: {}'.format(torch.cuda.is_available()))
    model = DeepJ().cuda()
    train(model, generator)

if __name__ == '__main__':
    main()
