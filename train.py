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

def train(model, data_generator):
    """
    Trains a model on multiple seq batches by iterating through a generator.
    """
    step = 1
    # Number of training steps per epoch
    epoch = 1
    epoch_len = 100
    # Keep tracks of all losses in each epoch
    all_losses = []
    total_loss = 0

    t = tqdm(data_generator, total=epoch_len)

    for data in t:
        loss = train_step(model, *data)
        total_loss += loss
        avg_loss = total_loss / step
        t.set_description('Epoch {}'.format(epoch))
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
            epoch += 1

        step += 1

def train_step(model, note_seq, replay_seq, beat_seq, style):
    """
    Trains the model on a single batch of sequence.
    """
    criterion = nn.BCELoss()
    # TODO: Clip gradient if needed.
    optimizer = optim.Adam(model.parameters())

    # Initialize hidden states
    states = model.init_states(BATCH_SIZE)

    # Zero out the gradient
    optimizer.zero_grad()

    loss = 0
    seq_len = note_seq.size()[0]
    # Iterate through the entire sequence
    for i in range(seq_len - 1):
        # TODO: We can apply custom input based on mistakes.
        targets = note_seq[:, i + 1]
        output, states = model(note_seq[:, i], targets, states)
        loss += criterion(output, targets)

    loss.backward()
    optimizer.step()
    return loss.data[0] / seq_len

def main():
    print('=== Dataset ===')
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Loading...')
    generator = batcher(sampler(load_styles()))
    print('=== Training ===')
    # print('GPU: {}'.format(torch.cuda.is_available()))
    model = DeepJ()#.cuda()
    train(model, generator)

if __name__ == '__main__':
    main()
