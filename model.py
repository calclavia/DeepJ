import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constants as const
from util import *
import numpy as np
import math
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend

class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_size = 4 * self.hidden_size

        # Weights
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.w_mih = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.w_mhh = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))

        # Biases
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))

        self.reset_parameters()
    
    def reset_parameters(self):
        stdev = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            param.data.normal_(-stdev, stdev)

    def forward(self, input, hidden):
        if hidden is None:
            hidden = torch.zeros((2, input.size(0), self.hidden_size), dtype=input.dtype, device=input.device)

        hx, cx = hidden

        if input.is_cuda:
            igates = F.linear(input, self.w_ih)
            m = F.linear(input, self.w_mih) * F.linear(hx, self.w_mhh)
            hgates = F.linear(m, self.w_hh)

            state = fusedBackend.LSTMFused.apply
            return state(igates, hgates, cx, self.b_ih, self.b_hh)
        
        m = F.linear(input, self.w_mih) * F.linear(hx, self.w_mhh)
        igates = F.linear(input, self.w_ih, self.b_ih) + F.linear(m, self.w_hh, self.b_hh)

        ingate = F.sigmoid(igates[:, :self.hidden_size])
        forgetgate = F.sigmoid(igates[:, self.hidden_size:self.hidden_size * 2])
        cellgate = F.tanh(igates[:, self.hidden_size * 2:self.hidden_size * 3])
        outgate = F.sigmoid(igates[:, -self.hidden_size:])
        
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
        return hy, cy

class DeepJ(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, num_units=const.NUM_UNITS, num_layers=3):
        super().__init__()
        self.num_units = num_units

        self.encoder = nn.Embedding(VOCAB_SIZE, num_units)
        self.decoder = nn.Linear(num_units, VOCAB_SIZE)

        # RNN
        self.rnns = nn.ModuleList([mLSTMCell(num_units, num_units) for _ in range(num_layers)])

    def forward_rnns(self, x, memory):
        if memory is None:
            memory = tuple(None for _ in self.rnns)

        new_memory = []

        for rnn, m in zip(self.rnns, memory):
            h, c = rnn(x, m)
            x = h
            new_memory.append(torch.stack((h, c), dim=0))

        return x, torch.stack(new_memory, dim=0)

    def forward(self, x, memory=None):
        seq_input = len(x.size()) == 2
        x = self.encoder(x)

        if seq_input:
            ys = []
            for t in range(x.size(1)):
                y, memory = self.forward_rnns(x[:, t], memory)
                ys.append(y)
            
            x = torch.stack(ys, dim=1)
        else:
            x, memory = self.forward_rnns(x, memory)

        hidden = x
        x = self.decoder(x)
        return x, memory, hidden

class Discriminator(nn.Module):
    def __init__(self, num_units=const.NUM_UNITS, num_layers=1):
        super().__init__()
        self.num_units = num_units

        # RNN
        self.rnn = nn.GRU(num_units, num_units, num_layers, batch_first=True, bidirectional=True)
        
        self.h1 = nn.Linear(num_units * 2, num_units)
        self.attn = nn.Linear(num_units, 1)
        self.h2 = nn.Linear(num_units, num_units)

        self.output = nn.Linear(num_units, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))

        # Self-attention over time
        attn_weights = torch.softmax(self.attn(x).float(), dim=1)
        x = (x.float() * attn_weights).sum(dim=1).to(x)

        x = self.output(x)
        return x.squeeze(-1)