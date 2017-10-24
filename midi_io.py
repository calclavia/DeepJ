"""
Handles MIDI file loading
"""
import mido
import numpy as np
import os
from constants import *
from util import *

# TODO: Handle custom tempo?
tempo = mido.bpm2tempo(120)

class TrackBuilder():
    def __init__(self, event_seq):
        self.event_seq = event_seq
        
        self.last_velocity = 0
        self.delta_time = 0
        self.tempo = tempo
        
        self.reset()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        evt = next(self.event_seq)

        # Interpret event data
        if evt >= VEL_OFFSET:
            # A velocity change
            self.last_velocity = (evt - VEL_OFFSET) * (MIDI_VELOCITY // VEL_QUANTIZATION)
        elif evt >= TIME_OFFSET:
            # Shifting forward in time
            quantized_time_shift = evt - TIME_OFFSET + 1
            assert quantized_time_shift >= 1 and quantized_time_shift <= TIME_QUANTIZATION
            time_shift = quantized_time_shift / TIME_QUANTIZATION * MAX_TIME_SHIFT
            self.delta_time += int(mido.second2tick(time_shift, self.midi_file.ticks_per_beat, self.tempo))
        elif evt >= NOTE_OFF_OFFSET:
            # Turning a note off
            note = evt - NOTE_OFF_OFFSET
            self.track.append(mido.Message('note_off', note=note, time=self.delta_time))
            self.delta_time = 0
        elif evt >= NOTE_ON_OFFSET:
            # Turning a note off
            note = evt - NOTE_ON_OFFSET
            self.track.append(mido.Message('note_on', note=note, time=self.delta_time, velocity=self.last_velocity))
            self.delta_time = 0

    def reset(self):
        self.midi_file = mido.MidiFile()
        self.track = mido.MidiTrack()
        self.track.append(mido.MetaMessage('set_tempo', tempo=self.tempo))
    
    def export(self):
        """
        Export buffer track to MIDI file
        """
        # Turn all notes off
        # for note in range(NUM_NOTES):
        #   self.track.append(mido.Message('note_off', note=note, time=0))

        # self.track.append(mido.Message('note_off', note=note, time=int(mido.second2tick(2, self.midi_file.ticks_per_beat, self.tempo))))
        self.midi_file.tracks.append(self.track)
        return_file = self.midi_file
        self.reset()
        return return_file

def seq_to_midi(event_seq):
    """
    Takes an event sequence and encodes it into MIDI file
    """
    track_builder = TrackBuilder(iter(event_seq))

    for _ in track_builder:
        pass

    return track_builder.export()

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
            # Delta time, in quantized units
            quantized_time = round(time_in_sec / MAX_TIME_SHIFT * TIME_QUANTIZATION)

            # Add in seconds
            while quantized_time > 0:
                time_add = min(quantized_time, TIME_QUANTIZATION)
                evt_index = TIME_OFFSET + time_add - 1
                assert evt_index >= TIME_OFFSET and evt_index < VEL_OFFSET
                events.append(evt_index)
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
                events.append(VEL_OFFSET + velocity)
                last_velocity = velocity

            events.append(NOTE_ON_OFFSET + msg.note)
        elif event_type == 'note_off':
            events.append(NOTE_OFF_OFFSET + msg.note)
    return np.array(events)

def load_midi(fname):
    cache_path = os.path.join(CACHE_DIR, fname + '.npy')
    try:
        seq = np.load(cache_path)
    except Exception as e:
        # Load
        mid = mido.MidiFile(fname)
        track = mido.merge_tracks(mid.tracks)
        seq = midi_to_seq(mid, track)

        # Perform caching
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, seq)
    return seq

def save_midi(fname, event_seq):
    """
    Takes a list of all notes generated per track and writes it to file
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    fpath = SAMPLES_DIR + '/' + fname + '.mid'
    midi_file = seq_to_midi(event_seq)
    print('Writing file', fpath)
    midi_file.save(fpath)
    
def save_midi_file(file, event_seq):
    """
    Takes a list of all notes generated per track and writes it to file
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    midi_file = seq_to_midi(event_seq)
    midi_file.save(file=file)

if __name__ == '__main__':
    # Test
    save_midi('midi_test1', load_midi('data/baroque/bach/bach_846.mid'))
    # save_midi('midi_test2', load_midi('data/classical/beethoven/appass_1.mid'))
    # save_midi('midi_test3', load_midi('data/jazz/Dannyboy.mid'))