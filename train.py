import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse

from dataset import *
from constants import *
from util import *
from model import DeepJ
from generate import generate, sample_note

def plot_loss(training_loss, validation_loss, name):
    # Draw graph
    plt.clf()
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.savefig(OUT_DIR + '/' + name)

def train(model, train_generator, train_len, val_generator, val_len, plot=True, gen_rate=1, patience=5):
    """
    Trains a model on multiple seq batches by iterating through a generator.
    """
    # Number of training steps per epoch
    epoch = 1
    total_step = 1

    # Keep tracks of all losses in each epoch
    train_losses = []
    val_losses = []

    # Epoch loop
    while True:
        # Training
        step = 1
        total_loss = 0

        t_gen = train_generator()
        t = tqdm(total=train_len)
        t.set_description('Epoch {}'.format(epoch))

        for data in t_gen:
            train_prob = max(MIN_SCHEDULE_PROB, 1 -SCHEDULE_RATE * total_step)
            loss = train_step(model, data, train_prob)

            total_loss += loss
            avg_loss = total_loss / step
            t.set_postfix(loss=avg_loss, prob=train_prob)
            t.update(BATCH_SIZE)

            step += 1
            total_step += 1
        t.close()
        train_losses.append(avg_loss)

        # Validation
        step = 1
        total_loss = 0

        v_gen = val_generator()
        t = tqdm(total=val_len)
        t.set_description('Validation {}'.format(epoch))

        for data in v_gen:
            loss = val_step(model, data)
            total_loss += loss
            avg_loss = total_loss / step
            t.set_postfix(loss=avg_loss)
            t.update(BATCH_SIZE)

            step += 1
        t.close()
        val_losses.append(avg_loss)

        if plot:
            plot_loss(train_losses, val_losses, 'loss.png')

        # Save model
        torch.save(model.state_dict(), OUT_DIR + '/model_' + str(epoch) + '.pt')

        # Generate
        if epoch % gen_rate:
            print('Generating...')
            generate(model, name='epoch_' + str(epoch))

        epoch += 1

        # Early stopping
        if epoch > patience:
            min_loss = min(val_losses)
            if min(val_losses[-patience:]) > min_loss:
                break

def train_step(model, data, teach_prob):
    """
    Trains the model on a single batch of sequence.
    """
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # Zero out the gradient
    optimizer.zero_grad()

    loss, avg_loss = compute_loss(model, data, teach_prob)

    loss.backward()
    optimizer.step()

    return avg_loss

def val_step(model, data):
    model.eval()
    return compute_loss(model, data, 1)[1]

def compute_loss(model, data, teach_prob):
    """
    Trains the model on a single batch of sequence.
    """
    note_seq, replay_seq, beat_seq, style = data
    criterion = nn.BCELoss()

    loss = 0
    seq_len = note_seq.size()[1]

    # Initialize hidden states
    states = None
    prev_note = note_seq[:, 0]

    # Iterate through the entire sequence
    for i in range(1, seq_len):
        beat = beat_seq[:, i]
        targets = note_seq[:, i]
        output, states = model(prev_note, beat, states, targets)
        loss += criterion(output, targets)

        # Choose note to feed based on coin flip (scheduled sampling)
        # TODO: Compare with and without scheduled sampling
        # TODO: Make sure this does not mess up gradients
        if np.random.random() <= teach_prob:
            prev_note = targets
        else:
            model.eval()
            prev_note, _ = sample_note(model, prev_note, beat, states, batch_size=BATCH_SIZE)
            prev_note = var(prev_note.data)
            model.train()

    return loss, loss.data[0] / seq_len

def main():
    parser = argparse.ArgumentParser(description='Trains model')
    parser.add_argument('--path', help='Load existing model?')
    parser.add_argument('--gen', default=1, type=int, help='Generate per how many epochs?')
    parser.add_argument('--noplot', default=False, action='store_true', help='Do not plot training/loss graphs')
    args = parser.parse_args()

    print('=== Loading Model ===')
    print('GPU: {}'.format(torch.cuda.is_available()))
    model = DeepJ()
    if torch.cuda.is_available():
        model.cuda()

    if args.path:
        model.load_state_dict(torch.load(args.path))
        print('Restored model from checkpoint.')
    print()

    print('=== Dataset ===')
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Loading data...')
    data = process(load_styles())
    print()
    print('Creating data generators...')
    train_ind, val_ind = validation_split(iteration_indices(data))
    train_generator = lambda: batcher(sampler(data, train_ind))
    val_generator = lambda: batcher(sampler(data, val_ind))
    print()

    print('=== Training ===')
    train(model, train_generator, len(train_ind), val_generator, \
         len(val_ind), plot=not args.noplot, gen=args.gen)

if __name__ == '__main__':
    main()
