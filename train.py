import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from dataset import *
from constants import *
from model import DeepJ
from generate import generate, sample_note

def train(model, data_generator):
    """
    Trains a model on multiple seq batches by iterating through a generator.
    """
    # Number of training steps per epoch
    step = 1
    epoch = 1
    epoch_len = 500
    total_step = 1

    # Keep tracks of all losses in each epoch
    all_losses = []
    total_loss = 0

    t = tqdm(total=epoch_len)

    for data in data_generator:
        t.set_description('Epoch {}'.format(epoch))

        train_prob = max(MIN_SCHEDULE_PROB, 1 -SCHEDULE_RATE * total_step)
        loss = train_step(model, train_prob, *data)

        total_loss += loss
        avg_loss = total_loss / step
        t.set_postfix(loss=avg_loss, prob=train_prob)
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
        total_step += 1

def train_step(model, teach_prob, note_seq, replay_seq, beat_seq, style):
    """
    Trains the model on a single batch of sequence.
    """
    model.train()
    criterion = nn.BCELoss()
    # TODO: Clip gradient if needed.
    optimizer = optim.Adam(model.parameters())

    # Zero out the gradient
    optimizer.zero_grad()

    loss = 0
    seq_len = note_seq.size()[1]

    # Initialize hidden states
    states = None
    prev_note = Variable(torch.zeros(note_seq[:, 0].size())).cuda()

    # Iterate through the entire sequence
    for i in range(seq_len):
        beat = beat_seq[:, i]
        targets = note_seq[:, i]
        output, states = model(prev_note, beat, states, targets)
        loss += criterion(output, targets)

        # TODO: Compare with and without scheduled sampling
        # TODO: Make sure this does not mess up gradients
        # Choose note to feed based on coin flip (scheduled sampling)
        if np.random.random() <= teach_prob:
            prev_note = targets
        else:
            model.eval()
            prev_note = sample_note(model, prev_note, beat, states, batch_size=BATCH_SIZE)
            prev_note = Variable(prev_note.data).cuda()
            model.train()

    loss.backward()
    optimizer.step()
    return loss.data[0] / seq_len

def main():
    print('=== Dataset ===')
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Loading...')
    generator = batcher(sampler(process(load_styles())))
    print()
    print('=== Training ===')
    print('GPU: {}'.format(torch.cuda.is_available()))
    model = DeepJ().cuda()
    train(model, generator)

if __name__ == '__main__':
    main()
