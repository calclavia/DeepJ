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
    def __init__(self, num_units=512, num_layers=3, style_units=32):
        super().__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.style_units = style_units

        # RNN
        self.rnns = [nn.LSTM(self.num_units, self.num_units, batch_first=True) for i in range(num_layers)]

        self.input_linear = nn.Linear(NUM_ACTIONS, self.num_units)
        self.output_linear = nn.Linear(self.num_units, NUM_ACTIONS)
        self.softmax = nn.Softmax()

        for i, rnn in enumerate(self.rnns):
            self.add_module('rnn_' + str(i), rnn)

        # Style
        self.style_linear = nn.Linear(NUM_STYLES, self.style_units)
        self.style_layers = [nn.Linear(self.style_units, self.num_units) for i in range(num_layers)]
        self.tanh = nn.Tanh()

        for i, layer in enumerate(self.style_layers):
            self.add_module('style_layers_' + str(i), layer)

    def forward(self, x, style, states=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        ## Process style ##
        # Distributed style representation
        style = self.style_linear(style)
        x = self.tanh(self.input_linear(x))

        ## Process RNN ##
        if states is None:
            states = [None for _ in range(self.num_layers)]

        for l, rnn in enumerate(self.rnns):
            # prev_x = x

            # Style integration
            style_activation = self.tanh(self.style_layers[l](style))
            style_seq = style_activation.unsqueeze(1)
            style_seq = style_seq.expand(batch_size, seq_len, self.num_units)
            x = x + style_seq

            x, states[l] = rnn(x, states[l])

            # Residual connection
            # if l != 0:
                # x = x + prev_x

        x = self.output_linear(x)
        return x, states

    def generate(self, x, style, states, temperature=1):
        """ Returns the probability of outputs """
        x, states = self.forward(x, style, states)
        seq_len = x.size(1)
        x = x.view(-1, NUM_ACTIONS)
        x = self.softmax(x / temperature)
        x = x.view(-1, seq_len, NUM_ACTIONS)
        return x, states
