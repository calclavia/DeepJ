import os
import math
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import argparse
import random

from dataset import *
from constants import *
from util import *
from model import DeepJ

criterion = nn.CrossEntropyLoss()

def plot_graph(training_loss, validation_loss, name):
    # Draw graph
    plt.clf()
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.savefig(OUT_DIR + '/' + name)

def train(args, model, train_loader, val_loader, optimizer, plot=True):
    """
    Trains a model on multiple seq batches by iterating through a generator.
    """
    # Number of training steps per epoch
    epoch = 1
    total_step = 1

    # Keep tracks of all losses in each epoch
    train_metrics = []
    val_metrics = []

    # Epoch loop
    while True:
        # Training
        step = 1
        total_metrics = 0

        with tqdm(train_loader) as t:
            t.set_description('Epoch {}'.format(epoch))
            
            for data in t:
                metrics = train_step(model, data, optimizer, total_step)

                total_metrics += metrics
                avg_metrics = total_metrics / step
                t.set_postfix(loss=avg_metrics[0])

                step += 1
                total_step += 1

        train_metrics.append(avg_metrics)

        # Validation
        step = 1
        total_metrics = 0

        with tqdm(val_loader) as t:
            t.set_description('Validation {}'.format(epoch))

            for data  in t:
                metrics = val_step(model, data, total_step)
                total_metrics += metrics
                avg_metrics = total_metrics / step
                t.set_postfix(loss=avg_metrics[0])

                step += 1
            
        val_metrics.append(avg_metrics)

        if plot:
            plot_graph([m[0] for m in train_metrics], [m[0] for m in val_metrics], 'loss.png')

        # Save model
        torch.save(model.state_dict(), OUT_DIR + '/model' + '.pt')

        epoch += 1

def train_step(model, data, optimizer, total_step):
    """
    Trains the model on a single batch of sequence.
    """
    model.train()

    loss, metrics = compute_metrics(model, data, total_step)
    
    # Scale the loss
    loss = loss * SCALE_FACTOR

    # Zero out the gradient
    model.zero_grad()
    loss.backward()
    param_copy = model.param_copy
    set_grad(param_copy, list(model.parameters()))

    # Unscale the gradient
    if SCALE_FACTOR != 1:
        for param in param_copy:
            param.grad.data /= SCALE_FACTOR

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    # Reference: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
    torch.nn.utils.clip_grad_norm_(param_copy, GRADIENT_CLIP)
    optimizer.step()

    # Copy the parameters back into the model
    copy_in_params(model, param_copy)
    return metrics

def val_step(model, data, total_step):
    model.eval()
    with torch.no_grad():
        return compute_metrics(model, data, total_step)[1]

def compute_metrics(model, data, total_step):
    """
    Trains the model on a single batch of sequence.
    """
    # Convert all tensors into variables
    seqs = data

    # Feed it to the model
    seqs = seqs.cuda()
    seq_inputs = seqs[:, :-1]
    seq_targets = seqs[:, 1:]

    logits, _ = model.forward_train(seq_inputs)

    # Compute the loss.
    # Note that we need to convert this back into a float because it is a large summation.
    # Otherwise, it will result in 0 gradient.
    loss = criterion(logits.contiguous().view(-1, VOCAB_SIZE).float(), seq_targets.contiguous().view(-1).data)
    return loss, np.array([loss.item()])

def main():
    parser = argparse.ArgumentParser(description='Trains model')
    parser.add_argument('--path', help='Load existing model?')
    parser.add_argument('--batch-size', default=256, type=int, help='Size of the batch')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--noplot', default=False, action='store_true', help='Do not plot training/loss graphs')
    parser.add_argument('--no-fp16', default=False, action='store_true', help='Disable FP16 training')
    args = parser.parse_args()
    args.fp16 = not args.no_fp16

    print('=== Loading Model ===')
    model = DeepJ().cuda()

    if args.fp16:
        # Wrap forward method
        # fwd = model.forward
        # model.forward = lambda x, states: fwd(x.half(), states)
        model.half()

    if args.path:
        model.load_state_dict(torch.load(args.path))
        print('Restored model from checkpoint.')

    # Construct optimizer
    param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model.parameters()]
    for param in param_copy:
        param.requires_grad = True
    optimizer = optim.Adam(param_copy, lr=args.lr, eps=1e-4)
    model.param_copy = param_copy

    params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])

    print('GPU: {}'.format(torch.cuda.is_available()))
    print('Batch Size: {}'.format(args.batch_size))
    print('FP16: {}'.format(args.fp16))
    print('# of Parameters: {}'.format(params))

    print()

    print('=== Dataset ===')
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Loading data...')
    train_loader, val_loader = get_tv_loaders(args)
    print()
    
    # Checks if training data sounds right.
    # from midi_io import save_midi
    # for i, seq in enumerate(train_batcher()[0]):
    #     save_midi('train_seq_{}'.format(i), seq.cpu().numpy())

    print('=== Training ===')
    train(args, model, train_loader, val_loader, optimizer, plot=not args.noplot)

if __name__ == '__main__':
    main()