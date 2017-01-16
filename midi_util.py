import midi
import numpy as np

DEFAULT_RES = 96
TICKS_PER_BEAT = 2
NUM_NOTES = 128

def midi_encode(composition,
                step=DEFAULT_RES // TICKS_PER_BEAT,
                resolution=TICKS_PER_BEAT):
    """
    Takes a composition array and encodes it into MIDI pattern
    """
    # Instantiate a MIDI Pattern (contains a list of tracks)
    pattern = midi.Pattern()
    pattern.resolution = resolution
    # Instantiate a MIDI Track (contains a list of MIDI events)
    track = midi.Track()
    # Append the track to the pattern
    pattern.append(track)

    velocity = 127

    # The current pattern being played
    current = np.zeros_like(composition[0])
    # Absolute tick of last event
    last_event_tick = 0
    # Amount of NOOP ticks
    noop_ticks = 0

    # track.append(midi.SetTempoEvent())
    for tick, data in enumerate(composition):
        data = np.array(data)

        if not np.array_equal(current, data):
            noop_ticks = 0

            # A bit difference vector.
            diff = data - current
            # TODO: Handle articulate
            for index, bit in np.ndenumerate(diff):
                if bit > 0:
                    # Was off, but now turned on
                    evt = midi.NoteOnEvent(
                        tick=(tick - last_event_tick) * step,
                        velocity=velocity,
                        pitch=index[0]
                    )
                    track.append(evt)
                    last_event_tick = tick
                elif bit < 0:
                    # Was on, but now turned off
                    evt = midi.NoteOffEvent(
                        tick=(tick - last_event_tick) * step,
                        pitch=index[0]
                    )
                    track.append(evt)
                    last_event_tick = tick
        else:
            noop_ticks += 1

        current = data

    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=noop_ticks)
    track.append(eot)
    return pattern


def midi_decode(pattern,
                classes=NUM_NOTES,
                track_index=0,
                step=DEFAULT_RES // TICKS_PER_BEAT):
    """
    Takes a MIDI pattern and decodes it into a composition array.
    """
    track = pattern[track_index]
    composition = [np.zeros((classes,))]

    for event in track:
        # Duplicate the last note pattern to wait for next event
        for _ in range(event.tick // step):
            composition.append(np.copy(composition[-1]))

        if isinstance(event, midi.EndOfTrackEvent):
            break

        # Modify the last note pattern
        if isinstance(event, midi.NoteOnEvent):
            pitch, velocity = event.data
            composition[-1][pitch] = 1

        if isinstance(event, midi.NoteOffEvent):
            pitch, velocity = event.data
            composition[-1][pitch] = 0
    return composition

import unittest


class TestMIDI(unittest.TestCase):

    def test_encode(self):
        composition = [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ]

        pattern = midi_encode(composition, step=1)
        self.assertEqual(pattern.resolution, 2)
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

        composition = midi_decode(pattern, 4)

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

        new_comp = midi_decode(midi_encode(composition, step=1), 4, step=1)
        np.testing.assert_array_equal(composition, new_comp)

if __name__ == '__main__':
    """
    # Test
    p = midi.read_midifile("out/Melody 001.mid")
    comp = midi_decode(p, track_index=1, step=1)
    p = midi_encode(comp, step=1, resolution=96)
    midi.write_midifile("out/Melody 002.mid", p)
    """
    unittest.main()
