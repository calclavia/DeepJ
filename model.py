import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constants as const
from util import *
import numpy as np
import math

class SRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w = nn.Parameter(torch.Tensor(self.hidden_size * 3, self.input_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size * 3))

        self.v_f = nn.Parameter(torch.Tensor(self.hidden_size))
        self.v_r = nn.Parameter(torch.Tensor(self.hidden_size))

        self.scaling = math.sqrt(3)
        self.reset_parameters()
    
    def reset_parameters(self):
        std = math.sqrt(3 / self.w.size(0))
        self.w.data.uniform_(-std, std)
        self.b.data.zero_()

        self.v_f.data.zero_()
        self.v_r.data.zero_()

    def forward(self, x, c=None):
        """
        Args:
            x: [batch, seq_len, input_size]
        """
        seq_input = len(x.size()) == 3
        batch_size = x.size(0)
        assert self.input_size == x.size(-1)

        wx = F.linear(x, self.w, self.b)
        wf_x = wx[..., :self.hidden_size]
        w_x = wx[..., self.hidden_size:self.hidden_size * 2]
        wr_x = wx[..., -self.hidden_size:]

        if c is None:
            c = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        
        if seq_input:
            all_cs = [c]

            for i in range(x.size(1)):
                c = self.recurrent(wf_x[:, i], w_x[:, i], c)
                all_cs.append(c)

            all_cs = torch.stack(all_cs, dim=1)

            h = self.highway(wr_x, x, all_cs[:, :-1], all_cs[:, 1:])
        else:
            prev_c = c
            c = self.recurrent(wf_x, w_x, c)

            h = self.highway(wr_x, x, prev_c, c)
        return h, c
    
    def recurrent(self, wf_x, w_x, c):
        f = torch.sigmoid(wf_x + self.v_f * c)
        c = f * (c - w_x) + w_x
        return c

    def highway(self, wr_x, x, prev_c, c):
        r = torch.sigmoid(wr_x + self.v_r * prev_c)
        x = x * self.scaling
        h = r * (c - x) + x
        return h

class DeepJ(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, num_units=const.NUM_UNITS, num_layers=8):
        super().__init__()
        self.num_units = num_units

        self.encoder = nn.Embedding(VOCAB_SIZE, num_units)
        self.decoder = nn.Linear(num_units, VOCAB_SIZE)

        # RNN
        self.rnns = nn.ModuleList([SRU(num_units, num_units) for _ in range(num_layers)])
   
    def forward_rnns(self, x, memory):
        if memory is None:
            memory = tuple(None for _ in self.rnns)

        new_memory = []

        for rnn, m in zip(self.rnns, memory):
            x, c = rnn(x, m)
            new_memory.append(c)

        return x, torch.stack(new_memory, dim=0)

    def forward(self, x, memory=None):
        x = self.encoder(x)

        x, memory = self.forward_rnns(x, memory)

        x = self.decoder(x)
        return x, memory