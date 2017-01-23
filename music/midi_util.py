import midi
from util import *

# TODO: Move all midi processing here
def midi_to_composition(pattern):
    """
    Converts a MIDI pattern to a composition array.
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
                composition.append(pitch)# - MIN_NOTE)
                composition += [NO_EVENT] * (event.tick // step)

        print(event)
    print(composition)

pattern2 = midi.read_midifile("data/Melody 001.mid")
midi_to_composition(pattern2)

# Instantiate a MIDI Pattern (contains a list of tracks)
pattern = midi.Pattern()
# Instantiate a MIDI Track (contains a list of MIDI events)
track = midi.Track()
# Append the track to the pattern
pattern.append(track)
pattern.append(pattern2[1])
# Instantiate a MIDI note on event, append it to the track
on = midi.NoteOnEvent(tick=0, velocity=100, pitch=midi.G_3)
track.append(on)
# Instantiate a MIDI note off event, append it to the track
off = midi.NoteOffEvent(tick=1000, pitch=midi.G_3)
track.append(off)
# Add the end of track event, append it to the track
eot = midi.EndOfTrackEvent(tick=1)
track.append(eot)
# Save the pattern to disk
midi.write_midifile("example.mid", pattern)
