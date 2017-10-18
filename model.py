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

    def forward(self, note_input, beat_in, states, condition_notes):
        out, states = self.time_axis(note_input, beat_in, states)
        out = self.note_axis(out, condition_notes)
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

        # Position + Pitchclass + Beat Context + Vicinity + Chord Context
        input_features = 1 + OCTAVE + BEAT_UNITS + VICINITY_UNITS + OCTAVE

        # Dropout
        self.input_dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()

        # Layers
        self.vicinity_conv = nn.Conv1d(NOTE_UNITS, VICINITY_UNITS, OCTAVE * 2 + 1, padding=OCTAVE)
        # self.chord_conv = nn.Conv1d(NOTE_UNITS, CHORD_UNITS, NUM_OCTAVES, dilation=OCTAVE, padding=OCTAVE)
        self.rnns = [nn.LSTMCell(input_features if i == 0 else num_units, num_units) for i in range(num_layers)]
        self.beat_proj = nn.Linear(NOTES_PER_BAR, BEAT_UNITS)

        # Constants
        self.pitch_pos = torch.range(0, self.num_notes - 1).unsqueeze(0) / self.num_notes

        stack_vecs = [one_hot(i % OCTAVE, OCTAVE) for i in range(self.num_notes)]
        stack_vecs = np.array(stack_vecs)
        self.pitch_class = torch.from_numpy(stack_vecs).float().unsqueeze(0)

        for i, rnn in enumerate(self.rnns):
            self.add_module('rnn_' + str(i), rnn)

    def compute_chord_context(self, notes_in):
        """
        The chord contex across all notes
        """
        # TODO: Convolution might make this obsolete.
        # Provides context of the chord for each note.
        batch_size = notes_in.size(0)
        # Normalize
        notes = notes_in[:, :, 0] / NUM_OCTAVES
        chord_context = notes.view(batch_size, OCTAVE, NUM_OCTAVES)
        chord_context = torch.sum(chord_context, 2).view(batch_size, 1, -1).repeat(1, self.num_notes, 1)
        return chord_context
        """
        # Permute the channels to 1st index
        notes = notes.permute(0, 2, 1)
        notes = self.chord_conv(notes)
        notes = self.tanh(notes)
        notes = self.dropout(notes)
        # Move the channel back
        notes = notes.permute(0, 2, 1)
        assert notes.size(1) == NUM_NOTES
        return notes
        """

    def compute_vicinity(self, notes):
        """
        The notes surrounding a given note.
        Performs convolution around a note's octave
        """
        """
        batch_size = notes.size(0)
        # Expand X with zero padding
        octave_padding = var(torch.zeros((batch_size, OCTAVE, notes.size(2))))
        pad_notes = torch.cat((octave_padding, notes, octave_padding), 1)

        # The notes an octave above and below
        vicinity = []

        # Pad each note
        for n in range(self.num_notes):
            vicinity.append(pad_notes[:, n:n + OCTAVE * 2 + 1, :])

        return torch.stack(vicinity, 1)
        """
        # Permute the channels to 1st index
        notes = notes.permute(0, 2, 1)
        notes = self.vicinity_conv(notes)
        notes = self.tanh(notes)
        notes = self.dropout(notes)
        # Move the channel back
        notes = notes.permute(0, 2, 1)
        return notes

    def forward(self, note_in, beat_in, states):
        """
        Args:
            input: batch_size x num_notes
        Return:
            ([batch_size, num_notes, features], states)
        """
        # Normalize TODO: Does this improve?
        # note_in = (note_in - 0.5) * 2
        # beat_in = (beat_in - 0.5) * 2

        batch_size = note_in.size()[0]
        notes = self.input_dropout(note_in)

        # Position of the note [batch_size x num_notes x 1]
        pitch_pos = var(self.pitch_pos).repeat(batch_size, 1).unsqueeze(2)
        # Pitch class of the note [batch_size x num_notes x OCTAVE]
        pitch_class = var(self.pitch_class).repeat(batch_size, 1, 1)

        # Beat context
        beat = self.input_dropout(beat_in)
        beat = self.beat_proj(beat)
        beat = self.dropout(beat)
        beat = beat.unsqueeze(1).repeat(1, self.num_notes, 1)

        # Vicinity
        vicinity = self.compute_vicinity(notes)

        # Chord context
        chord_context = self.compute_chord_context(note_in)
        chord_context = self.input_dropout(chord_context)

        features = torch.cat((pitch_pos, pitch_class, beat, vicinity, chord_context), 2)

        # Move all notes into batch dimension
        features = features.view(-1, features.size(2))

        # Initialize hidden states
        if not states:
            states = [[var(torch.zeros(features.size(0), self.num_units)) for _ in range(2)] for _ in self.rnns]

        out = features

        for l, rnn in enumerate(self.rnns):
            prev_out = out

            out, state = rnn(out, states[l])
            states[l] = (out, state)
            out = self.dropout(out)

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

        num_inputs = num_features + NOTE_UNITS

        self.input_dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.5)
        # TODO: LSTM Cell might be more efficient!
        self.rnn = nn.LSTM(num_inputs, num_units, num_layers, dropout=0.5, batch_first=True)
        self.output = nn.Linear(num_units, NOTE_UNITS)
        self.sigmoid = nn.Sigmoid()

    def forward(self, note_features, condition_notes, temperature=1):
        """
        Args:
            note_features: Features for each note [batch_size, num_notes, features]
            condition_notes: Target notes [batch_size, num_notes, note_units] (for training)
        """
        # Normalize TODO: Does this improve?
        # condition_notes = (condition_notes - 0.5) * 2
        batch_size = note_features.size()[0]

        note_features = self.dropout(note_features)
        condition_notes = self.input_dropout(condition_notes)

        # Used for the first target
        zero_padding = var(torch.zeros((batch_size, 1, NOTE_UNITS)))
        shifted = torch.cat((zero_padding, condition_notes), 1)[:, :-1]

        # Create note features
        note_features = torch.cat((note_features, shifted), 2)

        out, _ = self.rnn(note_features, None)

        out = self.dropout(out)

        # Apply output
        out = out.contiguous()
        out = out.view(-1, out.size(2))
        out = self.output(out)
        out = out.view(-1, self.num_notes, NOTE_UNITS)

        # Apply sigmoid to only probability outputs
        prob_out = self.sigmoid(out[:, :, 0:2] / temperature)
        return torch.cat((prob_out, out[:, :, 2:3]), 2)
