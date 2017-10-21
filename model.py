import torch
import torch.nn as nn
from torch.autograd import Variable
from constants import *
from util import *
import numpy as np

class DeepJ(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, num_units=512, num_layers=3):
        super().__init__()
        self.num_units = num_units
        self.num_layers = num_layers

        self.dropout = nn.Dropout(0.5)
        self.rnns = [nn.LSTMCell(NUM_ACTIONS if i == 0 else self.num_units, self.num_units) for i in range(num_layers)]
        self.output = nn.Linear(self.num_units, NUM_ACTIONS)
        self.softmax = nn.Softmax()

        for i, rnn in enumerate(self.rnns):
            self.add_module('rnn_' + str(i), rnn)

    def forward(self, x, states=None):
        batch_size = x.size(0)

        if states is None:
            states = [[var(torch.zeros(batch_size, self.num_units)) for _ in range(2)] for _ in range(self.num_layers)]

        for l, rnn in enumerate(self.rnns):
            prev_x = x

            x, state = rnn(x, states[l])
            states[l] = (x, state)
            x = self.dropout(x)

        x = self.output(x)
        return x, states

    def generate(self, inputs, states, temperature=1):
        x, states = self.forward(inputs, states)
        # TODO: Check temperature?
        x = self.softmax(x / temperature)

        # Sample action
        batch = []

        # Iterate over batches
        for prob in x.cpu().data.numpy():
            sampled_index = np.random.choice(len(prob), 1, p=prob)
            batch.append(one_hot(sampled_index, len(prob)))
        
        return np.array(batch), states
