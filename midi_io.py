"""
Handles MIDI file loading
"""
import mido
import numpy as np
import os
from constants import *
from util import *

# TODO: Handle custom tempo?
tempo = 400000

def seq_to_midi(event_seq):
    """
    Takes an event sequence and encodes it into MIDI file
    """
    midi_file = mido.MidiFile()
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    last_velocity = 0
    delta_time = 0
    i=0

    for evt in list(event_seq):
        index = np.argmax(evt)

        # Interpret event data
        if index >= VEL_OFFSET:
            # A velocity change
            last_velocity = (index - VEL_OFFSET) * (MIDI_VELOCITY // VEL_QUANTIZATION)
        elif index >= TIME_OFFSET:
            # Shifting forward in time
            time_shift = (index - TIME_OFFSET) / TIME_QUANTIZATION
            delta_time += int(mido.second2tick(time_shift, midi_file.ticks_per_beat, tempo))
        elif index >= NOTE_OFF_OFFSET:
            # Turning a note off
            note = index - NOTE_OFF_OFFSET
            track.append(mido.Message('note_off', note=note, time=delta_time))
            delta_time = 0
        elif index >= NOTE_ON_OFFSET:
            # Turning a note off
            note = index - NOTE_ON_OFFSET
            track.append(mido.Message('note_on', note=note, time=delta_time, velocity=last_velocity))
            delta_time = 0

    return midi_file

def midi_to_seq(midi_file, track):
    """
    Converts a MIDO track object into an event sequence
    """
    events = []
    last_velocity = None

    for msg in track:
        event_type = msg.type
        
        # Parse delta time
        if msg.time != 0:
            time_in_sec = mido.tick2second(msg.time, midi_file.ticks_per_beat, tempo)
            quantized_time = round(time_in_sec * TIME_QUANTIZATION)
            
            # Add in seconds
            while quantized_time > 0:
                time_add = min(quantized_time, MAX_TIME_SHIFT)
                events.append(one_hot(TIME_OFFSET + time_add, NUM_ACTIONS))
                quantized_time -= time_add

        # Ignore meta messages
        if msg.is_meta:
            continue

        # Ignore control changes
        if (event_type != 'note_on' and event_type != 'note_off'):
            continue

        velocity = msg.velocity // (MIDI_VELOCITY // VEL_QUANTIZATION)

        # Velocity = 0 is equivalent to note off
        if msg.type == 'note_on' and velocity == 0:
            event_type = 'note_off'

        if event_type == 'note_on':
            # See if we need to update velocity
            if last_velocity != velocity:
                events.append(one_hot(VEL_OFFSET + velocity, NUM_ACTIONS))
                last_velocity = velocity

            events.append(one_hot(NOTE_ON_OFFSET + msg.note, NUM_ACTIONS))
        elif event_type == 'note_off':
            events.append(one_hot(NOTE_OFF_OFFSET + msg.note, NUM_ACTIONS))
    return np.array(events)

def load_midi(fname):
    mid = mido.MidiFile(fname)
    track = mido.merge_tracks(mid.tracks)
    return midi_to_seq(mid, track)

def save_midi(fname, event_seq):
    """
    Takes a list of all notes generated per track and writes it to file
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    fpath = SAMPLES_DIR + '/' + fname + '.mid'
    midi_file = seq_to_midi(event_seq)
    print('Writing file', fpath)
    midi_file.save(fpath)

if __name__ == '__main__':
    # Test
    event_seq = load_midi('data/baroque/bach/bach_846.mid')
    save_midi('midi_test', event_seq)