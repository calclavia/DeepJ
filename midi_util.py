import midi
import numpy as np

def midi_encode(composition, resolution=2):
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
    last_event_tick = -1

    for tick, data in enumerate(composition):
        data = np.array(data)
        if not np.array_equal(current, data):
            if last_event_tick == -1:
                last_event_tick = 0

            # A bit difference vector.
            diff = data - current

            for index, bit in np.ndenumerate(diff):
                if bit > 0:
                    # Was off, but now turned on
                    evt = midi.NoteOnEvent(
                        tick=tick - last_event_tick,
                        velocity=velocity,
                        pitch=index
                    )
                    track.append(evt)
                    last_event_tick = tick
                elif bit < 0:
                    # Was on, but now turned off
                    evt = midi.NoteOffEvent(
                        tick=tick - last_event_tick,
                        pitch=index
                    )
                    track.append(evt)
                    last_event_tick = tick
        current = data

    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    return pattern


def midi_decode(pattern):
    """
    Takes a MIDI pattern and decodes it into a composition array.
    """
    # Extract first pattern
    # pattern.make_ticks_abs()
    step = pattern.resolution // TICKS_PER_BEAT
    track = pattern[1]
    # A list of notes currently on
    notes = []
    composition = []

    for event in track[1:-1]:
        # TODO: We only care about one note for now
        if isinstance(event, midi.NoteOnEvent) and len(notes) == 0:
            pitch, velocity = event.data
            notes.append(pitch)

        if isinstance(event, midi.NoteOffEvent):
            pitch, velocity = event.data
            if pitch in notes:
                notes.remove(pitch)
                # Write to composition
                composition.append(pitch)  # - MIN_NOTE)
                composition += [NO_EVENT] * (event.tick // step)

        print(event)
    print(composition)

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

        pattern = midi_encode(composition)
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
        self.assertEqual(on2.tick, 2)
        self.assertEqual(off1.tick, 2)
        self.assertEqual(off2.tick, 1)

    def test_decode(self):
        raise 'Not impl'

if __name__ == '__main__':
    unittest.main()

"""
Reference:
pattern2 = midi.read_midifile("data/Melody 001.mid")
midi_to_composition(pattern2)

"""
