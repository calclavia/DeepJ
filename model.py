import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from constants import *
from util import *

class DeepJ(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTMCell(NUM_ACTIONS, LSTM_UNITS)
        self.output = nn.Linear(LSTM_UNITS, NUM_ACTIONS)
        self.softmax = nn.Softmax()

    def forward(self, inputs, states=None):
        batch_size = inputs.size(0)

        if states is None:
            states = [[var(torch.zeros(batch_size, LSTM_UNITS)) for _ in range(2)]]

        x, state = self.rnn(inputs, states[0])
        states[0] = (x, state)

        x = self.output(x)
        return x, states

    def generate(self, inputs, states):
        x, states = self.forward(inputs, states)
        x = self.softmax(x)
        return x, states
