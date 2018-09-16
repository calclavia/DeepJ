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
from model import *
from generate import generate

from apex.fp16_utils import FP16_Optimizer

cce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCEWithLogitsLoss()

def plot_graph(training_loss, validation_loss, name):
    # Draw graph
    plt.clf()
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.savefig(OUT_DIR + '/' + name)

def train(args, model, d_model, train_loader, val_loader, optimizer, plot=True):
    """
    Trains a model on multiple seq batches by iterating through a generator.
    """
    model_save = model
    model = nn.DataParallel(model)
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
                metrics = train_step(model, d_model, data, optimizer, total_step)

                total_metrics += metrics
                avg_metrics = total_metrics / step
                t.set_postfix(mle_loss=avg_metrics[0], g_loss=avg_metrics[1], d_acc=avg_metrics[2])

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
        torch.save(model_save.state_dict(), OUT_DIR + '/model' + '.pt')

        epoch += 1

def train_step(model, d_model, data, optimizer, total_step):
    """
    Trains the model on a single batch of sequence.
    """
    model.train()

    # Convert all tensors into variables
    seqs = data

    # Create data sequences
    seqs = seqs.cuda()
    seq_inputs = seqs[:, :-1]
    seq_targets = seqs[:, 1:]

    ## Train the discriminator
    # Generate in teacher Forcing Mode
    logits, _, hidden_tf = model(seq_inputs)

    # Gereate in free-running mode
    _, hidden_fr = generate(model, start_token=seq_inputs[:, 0])
    hidden_fr = torch.stack(hidden_fr, dim=1)

    # Teacher forcing gets label 1. Free running gets label 0.
    d_inputs = torch.cat((hidden_fr.detach(), hidden_tf.detach()), dim=0)
    d_targets = torch.cat((torch.zeros(hidden_fr.size(0)), torch.ones(hidden_tf.size(0))), dim=0).long().cuda()

    d_logits = d_model(d_inputs).float()
    d_acc = ((d_logits > 0).long() == d_targets).float().mean()

    d_loss = bce_loss(d_logits, d_targets.float())

    # Back-propagate
    optimizer.zero_grad()
    optimizer.backward(d_loss)
    optimizer.clip_master_grads(GRADIENT_CLIP)
    optimizer.step()

    ## Train the generator
    # Freeze discriminator weights
    for p in d_model.parameters():
        p.requires_grad = False

    # Compute the loss.
    # Note that we need to convert this back into a float because it is a large summation.
    mle_loss = cce_loss(logits.contiguous().view(-1, VOCAB_SIZE).float(), seq_targets.contiguous().view(-1).data)
    
    # Fool the discriminator
    d_inputs = torch.cat((hidden_fr, hidden_tf), dim=0)

    d_logits = d_model(d_inputs).float()

    fr_d_preds = torch.sigmoid(d_logits[:hidden_fr.size(0)])
    tf_d_preds = torch.sigmoid(d_logits[-hidden_tf.size(0):])
    fool_loss = -torch.mean(torch.log(fr_d_preds) + torch.log(1 - tf_d_preds))

    g_loss = mle_loss + fool_loss

    # Back-propagate
    optimizer.zero_grad()
    optimizer.backward(g_loss)
    optimizer.clip_master_grads(GRADIENT_CLIP)
    optimizer.step()

    # Unfreeze discriminator weights
    for p in d_model.parameters():
        p.requires_grad = True

    return np.array([mle_loss.item(), g_loss.item(), d_acc.item()])

def val_step(model, data, total_step):
    with torch.no_grad():
        model.eval()

        # Convert all tensors into variables
        seqs = data

        # Feed it to the model
        seqs = seqs.cuda()
        seq_inputs = seqs[:, :-1]
        seq_targets = seqs[:, 1:]

        logits, *_ = model(seq_inputs)

        # Note that we need to convert this back into a float because it is a large summation.
        mle_loss = cce_loss(logits.contiguous().view(-1, VOCAB_SIZE).float(), seq_targets.contiguous().view(-1).data)
        return np.array([mle_loss.item()])

def main():
    parser = argparse.ArgumentParser(description='Trains model')
    parser.add_argument('--load', help='Load existing model?')
    parser.add_argument('--batch-size', default=64, type=int, help='Size of the batch')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')
    parser.add_argument('--noplot', default=False, action='store_true', help='Do not plot training/loss graphs')
    args = parser.parse_args()

    print('=== Loading Model ===')
    model = DeepJ().cuda().half()
    d_model = Discriminator().cuda().half()

    if args.load:
        model.load_state_dict(torch.load(args.load))
        print('Restored model from checkpoint.')

    # Construct optimizer
    optimizer = optim.Adam(list(model.parameters()) + list(d_model.parameters()), lr=args.lr)
    optimizer = FP16_Optimizer(optimizer, static_loss_scale=256)

    num_params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])

    print('GPU: {}'.format(torch.cuda.is_available()))
    print('Batch Size: {}'.format(args.batch_size))
    print('# of Parameters: {}'.format(num_params))

    print()

    print('=== Dataset ===')
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Loading data...')
    train_loader, val_loader = get_tv_loaders(args)
    print()
    
    # Outputs a training sample to check if training data sounds right.
    from midi_io import tokens_to_midi
    for i, seq in enumerate(train_loader):
        tokens_to_midi(seq[0].cpu().numpy()).save('out/train_seq_{}'.format(i))
        break

    print('=== Training ===')
    train(args, model, d_model, train_loader, val_loader, optimizer, plot=not args.noplot)

if __name__ == '__main__':
    main()