import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from constants import *
from util import *
import numpy as np

class DeepJ(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, num_units=512, num_layers=3, style_units=32):
        super().__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.style_units = style_units

        # RNN
        # self.rnns = [nn.LSTM((NUM_ACTIONS + style_units) if i == 0 else self.num_units, self.num_units, batch_first=True) for i in range(num_layers)]
        self.rnn = nn.LSTM(NUM_ACTIONS + style_units, self.num_units, num_layers, batch_first=True)

        self.output_linear = nn.Linear(self.num_units, NUM_ACTIONS)

        # for i, rnn in enumerate(self.rnns):
            # self.add_module('rnn_' + str(i), rnn)

        # Style
        self.style_linear = nn.Linear(NUM_STYLES, self.style_units)
        # self.style_layer = nn.Linear(self.style_units, self.num_units * self.num_layers)

    def forward(self, x, style, states=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Distributed style representation
        style = self.style_linear(style)
        # style = F.tanh(self.style_layer(style))
        style = style.unsqueeze(1).expand(batch_size, seq_len, self.style_units)
        x = torch.cat((x, style), dim=2)

        ## Process RNN ##
        # if states is None:
            # states = [None for _ in range(self.num_layers)]

        x, states = self.rnn(x, states)
        # for l, rnn in enumerate(self.rnns):
            # x, states[l] = rnn(x, states[l])
            # Style integration
            # x = x + style[:, l * self.num_units:(l + 1) * self.num_units].unsqueeze(1).expand(-1, seq_len, -1)

        x = self.output_linear(x)
        return x, states

    def generate(self, x, style, states, temperature=1):
        """ Returns the probability of outputs """
        x, states = self.forward(x, style, states)
        seq_len = x.size(1)
        x = x.view(-1, NUM_ACTIONS)
        x = F.softmax(x / temperature)
        x = x.view(-1, seq_len, NUM_ACTIONS)
        return x, states
