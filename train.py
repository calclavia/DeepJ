import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(model, data_generator):
    """
    Trains a model by iterating through a data_generator.
    """
    step = 1
    # Number of training steps per epoch
    epoch_len = 1000
    # Keep tracks of all losses in each epoch
    all_losses = []
    total_loss = 0

    t = tqdm(data_generator, total=epoch_len)

    for data in t:
        loss = model.fit(*data)
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
            torch.save({
                'epoch': len(all_losses),
                'state_dict': model.state_dict()
            }, 'out/checkpoint.tar' )

            step = 0

        step += 1
