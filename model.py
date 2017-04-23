import torch
import torch.nn as nn
from torch.autograd import Variable

class DeepJ(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

    def forward(self, input, state):
        pass

    def init_states(self):
        """
        Initializes the recurrent states
        """
        pass

    def train(self, input_seq, target_seq):
        """
        Trains the model on a single batch of sequences.
        """
        pass
