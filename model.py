import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from constants import *
from util import *
import numpy as np
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend

def fwd_mLSTMCell(hidden_size, input, hidden, w_ih, w_hh, w_mih, w_mhh, b_ih=None, b_hh=None):
    """
    mLSTMCell
    """
    hx, cx = hidden

    if input.is_cuda:
        igates = F.linear(input, w_ih)
        m = F.linear(input, w_mih) * F.linear(hx, w_mhh)
        hgates = F.linear(m, w_hh)
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, cx, b_ih, b_hh)

    m = F.linear(input, w_mih) * F.linear(hx, w_mhh)
    gates = F.linear(input, w_ih, b_ih) + F.linear(m, w_hh, b_hh)

    # Original implementation using chunk:
    # ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    # ingate = torch.sigmoid(ingate)
    # forgetgate = torch.sigmoid(forgetgate)
    # outgate = torch.sigmoid(outgate)
    # cellgate = torch.tanh(cellgate)

    # More efficient implementation
    sig_gates = torch.sigmoid(gates[:, :-hidden_size])
    cellgate = torch.tanh(gates[:, -hidden_size:])

    ingate = sig_gates[:, :hidden_size]
    forgetgate = sig_gates[:, hidden_size:hidden_size * 2]
    outgate = sig_gates[:, -hidden_size:]
    
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    return hy, cy

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
        nn.init.xavier_normal_(self.w_ih)
        nn.init.xavier_normal_(self.w_hh)
        nn.init.xavier_normal_(self.w_mih)
        nn.init.xavier_normal_(self.w_mhh)

        nn.init.zeros_(self.b_ih)
        nn.init.zeros_(self.b_hh)

    def forward(self, input, hidden):
        if hidden is None:
            hidden = torch.zeros((2, input.size(0), self.hidden_size), dtype=input.dtype, device=input.device)

        hy, cy = fwd_mLSTMCell(self.hidden_size, input, hidden, self.w_ih, self.w_hh, self.w_mih, self.w_mhh, self.b_ih, self.b_hh)
        return hy, torch.stack((hy, cy), dim=0)

class DeepJ(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, num_units=const.NUM_UNITS):
        super().__init__()
        self.num_units = num_units

        self.encoder = nn.Embedding(VOCAB_SIZE, num_units)
        # RNN
        self.rnn = mLSTMCell(num_units, num_units)

    def forward_train(self, x,  memory=None):
        assert len(x.size()) == 2
        x = self.encoder(x)

        ys = []
        for t in range(x.size(1)):
            y, memory = self.rnn(x[:, t], memory)
            ys.append(y)
        
        x = torch.stack(ys, dim=1)

        x = self.decoder(x)
        return x, memory
    
    def decoder(self, x):
        # Decoder and encoder weights are tied
        return F.linear(x, self.encoder.weight)

    def forward(self, x, memory=None, temperature=1):
        """ Returns the probability of outputs.  """
        assert len(x.size()) == 1

        x = self.encoder(x)
        x, memory = self.rnn(x, memory)
        x = self.decoder(x)

        x = F.softmax(x / temperature, dim=-1)
        return x, memory
