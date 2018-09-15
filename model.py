import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constants as const
from util import *
import numpy as np
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend

class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b

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

        # self.ln_gate = LayerNorm(self.gate_size)
        # self.ln_out = LayerNorm(self.hidden_size)

        self.act = torch.tanh#F.relu

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

        hx, cx = hidden
        m = F.linear(input, self.w_mih) * F.linear(hx, self.w_mhh)
        gates = F.linear(input, self.w_ih, self.b_ih) + F.linear(m, self.w_hh, self.b_hh)

        # TODO: Layer norm should be applied separately per gate
        # gates = self.ln_gate(gates)

        sig_gates = torch.sigmoid(gates[:, :-self.hidden_size])
        cellgate = self.act(gates[:, -self.hidden_size:])
        ingate = sig_gates[:, :self.hidden_size]
        forgetgate = sig_gates[:, self.hidden_size:self.hidden_size * 2]
        outgate = sig_gates[:, -self.hidden_size:]

        cy = (forgetgate * cx) + (ingate * cellgate)
        # hy = outgate * self.act(self.ln_out(cy))
        hy = outgate * self.act(cy)
        return hy, torch.stack((hy, cy), dim=0)

class DeepJ(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, num_units=const.NUM_UNITS):
        super().__init__()
        self.num_units = num_units

        self.encoder = nn.Embedding(VOCAB_SIZE, num_units)
        self.decoder = nn.Linear(num_units, VOCAB_SIZE)

        # RNN
        self.rnn = mLSTMCell(num_units, num_units)

    # def decoder(self, x):
    #     # Decoder and encoder weights are tied
    #     return F.linear(x, self.encoder.weight)

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

    def forward(self, x, memory=None, temperature=1):
        """ Returns the probability of outputs.  """
        assert len(x.size()) == 1

        x = self.encoder(x)
        x, memory = self.rnn(x, memory)
        x = self.decoder(x)

        x = torch.softmax(x / temperature, dim=-1)
        return x, memory
