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

    def forward(self, note_input, states, targets=None):
        out, states = self.time_axis(note_input, states)
        out = self.note_axis(out, targets)
        return out, states

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
        input_features = 1 + OCTAVE + (OCTAVE * 2 + 1) + OCTAVE

        self.input_dropout = nn.Dropout(0.2)
        self.rnn = nn.LSTM(input_features, num_units, num_layers, dropout=0.5)

        # Constants
        self.pitch_pos = torch.range(0, self.num_notes - 1).unsqueeze(0) / self.num_notes
        self.pitch_class = torch.stack([torch.from_numpy(one_hot(i % OCTAVE, OCTAVE)) for i in range(OCTAVE)]).unsqueeze(0)

    def forward(self, note_in, states):
        """
        Args:
            input: batch_size x num_notes
        Return:
            ([batch_size, num_notes, features], states)
        """
        batch_size = note_in.size()[0]
        notes = self.input_dropout(note_in)

        # Position of the note [batch_size x num_notes]
        pitch_pos = Variable(self.pitch_pos).cuda().repeat(batch_size, 1).unsqueeze(2)
        # Pitch class of the note [batch_size x num_notes x OCTAVE]
        pitch_class = Variable(self.pitch_class).cuda().repeat(batch_size, NUM_OCTAVES, 1).float()

        # Provides context of the chord for each note.
        # TODO: Should this be relative to the note?
        chord_context = notes.view(batch_size, OCTAVE, NUM_OCTAVES)
        chord_context = torch.sum(chord_context, 2).view(batch_size, 1, -1).repeat(1, self.num_notes, 1)

        # Expand X with zero padding
        octave_padding = Variable(torch.zeros((batch_size, OCTAVE))).cuda()
        pad_notes = torch.cat((octave_padding, notes, octave_padding), 1)

        # The notes an octave above and below
        vicinity = []

        for n in range(self.num_notes):
            vicinity.append(pad_notes[:, n:n + OCTAVE * 2 + 1])

        vicinity = torch.stack(vicinity, 1)

        features = torch.cat((pitch_pos, pitch_class, vicinity, chord_context), 2)

        # Move all notes into batch dimension
        features = features.view(1, -1, features.size(2))
        out, states = self.rnn(features, states)

        out = out.view(batch_size, self.num_notes, -1)
        return out, states

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
        self.rnn = nn.LSTM(num_inputs, num_units, num_layers, dropout=0.5, batch_first=True)
        self.output = nn.Linear(num_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, note_features, targets=None):
        """
        Args:
            note_features: Features for each note [batch_size, num_notes, features]
            targets: Target notes [batch_size, num_notes] (for training)
        """
        batch_size = note_features.size()[0]

        if targets:
            targets = self.input_dropout(targets)

            # Used for the first target
            zero_padding = Variable(torch.zeros((batch_size, 1))).cuda()
            shifted = torch.cat((zero_padding, targets), 1)[:, :-1]
            shifted = shifted.unsqueeze(2)

            # Create note features
            note_features = torch.cat((note_features, shifted), 2)

            out, _ = self.rnn(note_features, None)
        else:
            raise 'Implement generation logic'
            """
            # For generation?
            # Note axis hidden state
            state = None

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
            """

        # Apply output
        out = self.output(out.view(-1, out.size(2)))
        out = self.sigmoid(out)
        return out
