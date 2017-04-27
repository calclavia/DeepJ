import torch
from torch.autograd import Variable
from midi_util import *
from model import DeepJ
from util import *
import unittest

class TestModel(unittest.TestCase):
    def test_pitch_pos(self):
        model = DeepJ(3)
        pitch_pos = model.time_axis.pitch_pos
        self.assertEqual(pitch_pos.size(), (1, 3))
        np.testing.assert_allclose(pitch_pos.numpy(), [[0, 1/3, 2/3]])

    def test_pitch_class(self):
        model = DeepJ(3)
        pitch_class = model.time_axis.pitch_class
        self.assertEqual(pitch_class.size(), (1, 3, OCTAVE))
        np.testing.assert_allclose(pitch_class.numpy(), [[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]])
    """
    def test_chord_context(self):
        model = DeepJ(3)
        test_input = torch.FloatTensor([[1, 1, 0]])
        context = model.time_axis.compute_chord_context(test_input)

        self.assertEqual(context.size(), (1, 3, OCTAVE))
        np.testing.assert_allclose(context.numpy(), [[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]])
    """

    def test_vicinity(self):
        model = DeepJ(3)
        test_input = var(torch.FloatTensor([[1, 1, 0]]))
        vicinity = model.time_axis.compute_vicinity(test_input)
        self.assertEqual(vicinity.size(), (1, 3, OCTAVE * 2 + 1))

        empty_octave = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        np.testing.assert_allclose(vicinity.cpu().data.numpy(), [[
            empty_octave + [1] + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [1] + empty_octave,
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1] + [0] + empty_octave
        ]])

class TestMIDIUtil(unittest.TestCase):

    def test_encode(self):
        composition = [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ]

        articulation = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]

        pattern = midi_encode(composition, articulation, step=1)
        self.assertEqual(pattern.resolution, NOTES_PER_BEAT)
        self.assertEqual(len(pattern), 1)
        track = pattern[0]
        self.assertEqual(len(track), 4 + 1)
        on1, on2, off1, off2 = track[:-1]
        self.assertIsInstance(on1, midi.NoteOnEvent)
        self.assertIsInstance(on2, midi.NoteOnEvent)
        self.assertIsInstance(off1, midi.NoteOffEvent)
        self.assertIsInstance(off2, midi.NoteOffEvent)

        self.assertEqual(on1.tick, 0)
        self.assertEqual(on1.pitch, 1)
        self.assertEqual(on2.tick, 2)
        self.assertEqual(on2.pitch, 3)
        self.assertEqual(off1.tick, 2)
        self.assertEqual(off1.pitch, 1)
        self.assertEqual(off2.tick, 1)
        self.assertEqual(off2.pitch, 3)

    def test_decode(self):
        # Instantiate a MIDI Pattern (contains a list of tracks)
        pattern = midi.Pattern(resolution=96)
        # Instantiate a MIDI Track (contains a list of MIDI events)
        track = midi.Track()
        # Append the track to the pattern
        pattern.append(track)

        track.append(midi.NoteOnEvent(tick=0, velocity=127, pitch=0))
        track.append(midi.NoteOnEvent(tick=96, velocity=127, pitch=1))
        track.append(midi.NoteOffEvent(tick=0, velocity=127, pitch=0))
        track.append(midi.NoteOffEvent(tick=48, velocity=127, pitch=1))
        track.append(midi.EndOfTrackEvent(tick=1))

        composition, articulation = midi_decode(pattern, 4, step=DEFAULT_RES // 2)

        np.testing.assert_array_equal(composition, [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ])

    def test_encode_decode(self):
        composition = [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ]

        articulation = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]

        new_comp, new_artic = midi_decode(midi_encode(composition, articulation, step=1), 4, step=1)
        np.testing.assert_array_equal(composition, new_comp)

    def test_articulation_decode(self):
        # Instantiate a MIDI Pattern (contains a list of tracks)
        pattern = midi.Pattern(resolution=96)
        # Instantiate a MIDI Track (contains a list of MIDI events)
        track = midi.Track()
        # Append the track to the pattern
        pattern.append(track)

        track.append(midi.NoteOnEvent(tick=0, velocity=127, pitch=1))
        track.append(midi.NoteOnEvent(tick=0, velocity=127, pitch=3))
        track.append(midi.NoteOffEvent(tick=1, velocity=127, pitch=1))
        track.append(midi.NoteOnEvent(tick=2, velocity=127, pitch=1))
        track.append(midi.NoteOnEvent(tick=2, velocity=127, pitch=3))
        track.append(midi.EndOfTrackEvent(tick=1))

        composition, articulation = midi_decode(pattern, 4, step=3)

        np.testing.assert_array_equal(articulation, [
            [0., 0., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 0., 0.]
        ])

    def test_articulation_encode_decode(self):
        composition = [
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 0]
        ]

        articulation = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 0]
        ]

        new_comp, new_artic = midi_decode(midi_encode(composition, articulation, step=2), 4, step=2)
        np.testing.assert_array_equal(composition, new_comp)
        np.testing.assert_array_equal(articulation, new_artic)

unittest.main()
