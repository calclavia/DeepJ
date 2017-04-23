import torch
import torch.nn as nn
from torch.autograd import Variable

class DeepJ(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, num_notes):
        super().__init__()
        self.num_notes = num_notes
        self.time_axis = TimeAxis()
        self.note_axis = NoteAxis()

    def forward(self, input, state):
        pass

    def init_states(self, batch_size):
        """
        Initializes the recurrent states
        """
        pass
