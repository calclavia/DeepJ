import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constants as const
from util import *
import numpy as np
import math
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_size = 4 * self.hidden_size

        # Weights
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))

        # Biases
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))

        self.reset_parameters()
    
    def reset_parameters(self):
        stdev = math.sqrt(1.0 / self.hidden_size)

        self.w_ih.data.normal_(0, stdev)
        self.w_hh.data.normal_(0, stdev)
        
        self.b_ih.data.zero_()
        self.b_hh.data.zero_()

    def forward(self, x, hidden):
        if hidden is None:
            hidden = torch.zeros((2, x.size(0), self.hidden_size), dtype=x.dtype, device=x.device)

        hx, cx = hidden

        if x.is_cuda:
            igates = F.linear(x, self.w_ih)
            hgates = F.linear(hx, self.w_hh)

            state = fusedBackend.LSTMFused.apply
            return state(igates, hgates, cx, self.b_ih, self.b_hh)
        
        igates = F.linear(x, self.w_ih, self.b_ih) + F.linear(hx, self.w_hh, self.b_hh)

        ingate = torch.sigmoid(igates[:, :self.hidden_size])
        forgetgate = torch.sigmoid(igates[:, self.hidden_size:self.hidden_size * 2])
        cellgate = torch.tanh(igates[:, self.hidden_size * 2:self.hidden_size * 3])
        outgate = torch.sigmoid(igates[:, -self.hidden_size:])

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        return hy, cy

class DeepJ(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, num_units=const.NUM_UNITS, num_seqs=const.NUM_SEQS, num_layers=3):
        super().__init__()
        self.num_units = num_units
        self.style_units = round(math.sqrt(num_seqs))

        self.encoder = nn.Embedding(VOCAB_SIZE, num_units)
        self.decoder = nn.Linear(num_units, VOCAB_SIZE)

        self.style_encoder = nn.Embedding(num_seqs, self.style_units)
        self.style_linear = nn.Linear(self.style_units, num_units)

        # RNN
        self.rnns = nn.ModuleList([LSTMCell(num_units, num_units) for _ in range(num_layers)])

    def forward_rnns(self, x, memory):
        if memory is None:
            memory = tuple(None for _ in self.rnns)

        new_memory = []

        for rnn, m in zip(self.rnns, memory):
            x, c = rnn(x, m)
            new_memory.append(torch.stack((x, c), dim=0))

        return x, torch.stack(new_memory, dim=0)

    def forward(self, x, seq_id, memory=None):
        seq_input = len(x.size()) == 2
        x = self.encoder(x)

        if seq_id.dtype == torch.LongTensor or seq_id.dtype == torch.cuda.LongTensor:
            # Use embedding matrix. Otherwise, it's already embedded.
            seq_id = self.style_encoder(seq_id)
        
        style_x = self.style_linear(seq_id)

        if seq_input:
            # Broadcast across sequence
            style_x = style_x.unsqueeze(1)

        x = x + style_x

        if seq_input:
            ys = []
            for t in range(x.size(1)):
                y, memory = self.forward_rnns(x[:, t], memory)
                ys.append(y)
            
            x = torch.stack(ys, dim=1)
        else:
            x, memory = self.forward_rnns(x, memory)

        x = self.decoder(x)
        return x, memory
    
    def compute_style(self, seq_ids):
        """
        Computes the style embedding vector as a mixture of sequences.
        Args:
            seq_ids: [Batch, Num]
        """
        return self.style_encoder(seq_ids).mean(dim=1)