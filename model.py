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
    def __init__(self, num_units=512, num_layers=5, style_units=32):
        super().__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.style_units = style_units

        self.embedding = nn.Embedding(NUM_ACTIONS, num_units)

        # RNN
        self.rnn = nn.GRU(num_units + style_units, num_units, num_layers, batch_first=True)

        self.output_linear = nn.Linear(self.num_units, NUM_ACTIONS)

        # Style
        self.style_linear = nn.Linear(NUM_STYLES, self.style_units)

    def forward(self, x, style, memory=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Distributed style representation
        style = self.style_linear(style)
        style = style.unsqueeze(1).expand(batch_size, seq_len, self.style_units)
        x = self.embedding(x)
        x = torch.cat((x, style), dim=2)

        x, memory = self.rnn(x, memory)

        x = self.output_linear(x)
        return x, memory

    def generate(self, x, style, memory, temperature=1):
        """ Returns the probability of outputs """
        x, memory = self.forward(x, style, memory)
        seq_len = x.size(1)
        x = x.view(-1, NUM_ACTIONS)
        x = F.softmax(x / temperature, dim=1)
        x = x.view(-1, seq_len, NUM_ACTIONS)
        return x, memory
