import torch
import torch.nn as nn
from torch.autograd import Variable
from constants import *
from util import *

class DeepJ(nn.Module):
    """
    The DeepJ neural network model architecture.
    """
    def __init__(self, num_notes=NUM_NOTES):
        super().__init__()
        self.num_notes = num_notes
        self.time_axis = TimeAxis(num_notes, TIME_AXIS_UNITS, 2)
        self.note_axis = NoteAxis(num_notes, TIME_AXIS_UNITS, NOTE_AXIS_UNITS, 2)

    def forward(self, note_input, targets, states):
        out, states = self.time_axis(note_input, states)
        out = self.note_axis(out, targets)
        return out, states

    def init_states(self, batch_size):
        """
        Initializes the recurrent states
        """
        return self.time_axis.init_states(batch_size)

class TimeAxis(nn.Module):
    """
    Time axis module that learns temporal patterns.
    """
    def __init__(self, num_notes, num_units, num_layers):
        super().__init__()
        self.num_notes = num_notes
        self.num_units = num_units
        self.num_layers = num_layers

        # Position + Pitchclass + Vicinity + Chord Context
        input_features = 1 + OCTAVE + (OCTAVE * 2 + 1)
        # input_features = 1 + OCTAVE + (OCTAVE * 2 + 1) + OCTAVE

        self.input_dropout = nn.Dropout(0.2)
        self.rnn = nn.LSTM(input_features, num_units, num_layers, dropout=0.5)

    def forward(self, note_in, states):
        """
        Args:
            input: batch_size x num_notes
        Return:
            ([batch_size, num_notes, features], states)
        """
        batch_size = note_in.size()[0]
        x = note_in
        x = self.input_dropout(x)

        ## Constants
        # Position of the note [batch_size x num_notes]
        pitch_pos = Variable(torch.range(0, self.num_notes - 1) / self.num_notes).cuda()
        pitch_pos = pitch_pos.unsqueeze(0).repeat(batch_size, 1).unsqueeze(2)
        # Pitch class of the note [batch_size x num_notes x OCTAVE]
        pitch_class = Variable(torch.stack([torch.from_numpy(one_hot(i % OCTAVE, OCTAVE)) for i in range(OCTAVE)])).cuda()
        pitch_class = pitch_class.unsqueeze(0).repeat(batch_size, NUM_OCTAVES, 1).float()

        chord_context = x.view(batch_size, OCTAVE, NUM_OCTAVES)
        chord_context = torch.sum(chord_context, 2).squeeze()

        # Expand X with zero padding
        octave_padding = Variable(torch.zeros((batch_size, OCTAVE))).cuda()
        x = torch.cat((octave_padding, x, octave_padding), 1)

        outs = []
        next_states = []

        # Every note feeds through the shared RNN
        for n in range(self.num_notes):
            pitch_pos_in = pitch_pos[:, n, :]
            pitch_class_in = pitch_class[:, n, :]
            vicinity = x[:, n:n + OCTAVE * 2 + 1]

            rnn_in = torch.cat((pitch_pos_in, pitch_class_in, vicinity), 1)
            # rnn_in = torch.cat((pitch_pos, pitch_class, vicinity, chord_context), 1)
            rnn_in = rnn_in.view(1, batch_size, -1)
            out, state = self.rnn(rnn_in, states[n])
            outs.append(out)
            next_states.append(state)

        x = torch.cat(outs)
        x = x.view(batch_size, self.num_notes, -1)
        return x, next_states

    def init_states(self, batch_size):
        """
        Initializes the recurrent states
        """
        return [(Variable(torch.zeros(self.num_layers, batch_size, self.num_units)).cuda(),
                Variable(torch.zeros(self.num_layers, batch_size, self.num_units)).cuda())
                for i in range(self.num_notes)]

class NoteAxis(nn.Module):
    """
    Note axis module that learns conditional note generation.
    """
    def __init__(self, num_notes, num_features, num_units, num_layers):
        super().__init__()
        self.num_notes = num_notes
        self.num_units = num_units
        self.num_layers = num_layers

        num_inputs = num_features + 1

        self.input_dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.5)
        self.rnn = nn.LSTM(num_inputs, num_units, num_layers, dropout=0.5)
        self.output = nn.Linear(num_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, note_features, targets):
        """
        Args:
            note_features: Features for each note [batch_size, num_notes, features]
            targets: Target notes [batch_size, num_notes] (for training)
        """
        batch_size = note_features.size()[0]
        # TODO: Somehow these numbers are greater than one...
        targets = self.input_dropout(targets)
        # Used for the first target
        zero_target = Variable(torch.zeros((batch_size, 1))).cuda()

        # Note axis hidden state
        state = (Variable(torch.zeros(self.num_layers, batch_size, self.num_units)).cuda(),
                Variable(torch.zeros(self.num_layers, batch_size, self.num_units)).cuda())

        outs = []

        for n in range(self.num_notes):
            # Slice out the current note's feature
            feature_in = note_features[:, n, :]
            condition_in = targets[:, n - 1].unsqueeze(1) if n > 0 else zero_target

            rnn_in = torch.cat((feature_in, condition_in), 1)
            rnn_in = rnn_in.view(1, batch_size, -1)
            out, state = self.rnn(rnn_in, state)

            # Make a probability prediction
            out = out.squeeze(0)
            out = self.output(out)
            out = self.sigmoid(out)
            outs.append(out)

        out = torch.cat(outs, 1)
        return out
