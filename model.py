import torch
import torch.nn as nn
from torch.autograd import Variable
from constants import *
from util import *

class DeepJ(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, num_notes):
        super().__init__()
        self.num_notes = num_notes
        self.time_axis = TimeAxis(num_notes)
        self.note_axis = NoteAxis()

    def forward(self, input, state):
        pass

    def init_states(self, batch_size):
        """
        Initializes the recurrent states
        """
        # TODO: CUDA
        pass

class TimeAxis(nn.Module):
    """
    Time axis module that learns temporal patterns.
    """
    def __init__(self, num_notes, num_units=256, num_layers=2):
        super().__init__()
        self.num_notes = num_notes

        # Position + Pitchclass + Vicinity + Chord Context
        input_features = 1 + OCTAVE + (OCTAVE * 2 + 1) + OCTAVE

        self.input_dropout = nn.Dropout(0.2)
        self.rnn = nn.LSTM(num_notes, num_units, num_layers, dropout=0.5)

        ## Constants
        # Position of the note [1 x num_notes]
        self.pitch_pos = torch.arange(0, num_notes).unsqueeze(0) / num_notes
        assert self.pitch_pos.size() == (1, num_notes)

        # Pitch class of the note [1 x num_notes x OCTAVE]
        self.pitch_class = torch.stack([one_hot(i, OCTAVE) for i in range(OCTAVE)]) \
                            .repeat(NUM_OCTAVE, 1) \
                            .unsqueeze(0)  / OCTAVE
        assert self.pitch_class.size() == (1, num_notes, OCTAVE)

    def forward(self, note_in, state):
        """
        Args:
            input: batch_size x num_notes
        Return:
            ([batch_size, num_notes, features], states)
        """
        batch_size = note_in.size()[0]
        x = note_in
        x = self.input_dropout(x)

        batch_pitch_pos = self.pitch_pos.repeat(batch_size, 1)
        batch_pitch_class = self.pitch_class.repeat(batch_size, 1, 1)
        batch_chord_context = note_in.view(batch_size, OCTAVE, NUM_OCTAVE)
        chord_context = torch.sum(batch_chord_context, 2).unsqueeze(2)

        outs = []
        states = []

        # Every note feeds through the shared RNN
        for n in range(self.num_notes):
            pitch_pos = batch_pitch_pos[:, n].unsqueeze(1)
            pitch_class = batch_pitch_class[:, n, :]
            vicinity = x[:, n].unsqueeze(1)

            rnn_input = torch.cat((pitch_pos, pitch_class, vicinity, chord_context), 1)
            rnn_input = rnn_input.view(1, batch_size, -1)
            out, state = self.rnn(rnn_input, state[n])

            outs.append(out)
            states.append(state)

        x = torch.stack(outs, dim=1)
        return x, states

class NoteAxis(nn.Module):
    """
    Note axis module that learns conditional note generation.
    """
    pass
