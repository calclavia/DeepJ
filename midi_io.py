"""
Handles MIDI file loading
"""
import mido
import math
import numpy as np
import os
from constants import *
from util import *
import subprocess
import urllib.request

def tokens_to_midi(tokens):
    """
    Takes an event sequence and encodes it into MIDI file
    """
    midi_file = mido.MidiFile()
    midi_file.ticks_per_beat = TICKS_PER_BEAT
    track = mido.MidiTrack()

    tempo = mido.bpm2tempo(120)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    # TODO: Allow flexible starting note
    cur_note = 36
    # TODO: Introduce velocity
    last_velocity = 100
    delta_time = 0
    
    for token in tokens:
        token_type = TOKEN_IDS[token]
        # delta_time in mido ticks
        ticks = delta_time

        if token_type == 'wait':
            delta_time += 1
        elif token_type == 'note_inc':
            cur_note = min(cur_note + 1, NUM_NOTES)
        elif token_type == 'note_dec':
            cur_note = max(cur_note - 1, 0)
        elif token_type == 'note_on':
            track.append(mido.Message('note_on', note=cur_note, time=ticks, velocity=last_velocity))
            delta_time = 0  
        elif token_type == 'note_off':
            track.append(mido.Message('note_off', note=cur_note, time=ticks))
            delta_time = 0
        else:
            raise Exception('Invalid token', token)

    midi_file.tracks.append(track)
    return midi_file

def midi_to_tokens(midi_file, track):
    """
    Converts a MIDO track object into a raw string representation
    """
    tokens = []
    tempo = None
    last_note = None
    last_velocity = None
    
    # TODO: Reorder notes for least sequence length. Low -> High
    for msg in track:
        event_type = msg.type
        
        # Parse delta time
        if msg.time != 0:
            # Convert into our ticks representation
            ticks = int((msg.time / midi_file.ticks_per_beat) * TICKS_PER_BEAT)
            tokens += [TOKEN_IDS.index('wait')] * ticks

        # Ignore meta messages
        if msg.is_meta:
            if msg.type == 'set_tempo':
                # Handle tempo setting
                tempo = msg.tempo
            continue

        # Ignore control changes
        if event_type != 'note_on' and event_type != 'note_off':
            continue

        if last_note is None:
            last_note = msg.note

        note_diff = msg.note - last_note
        last_note = msg.note
        tokens += [TOKEN_IDS.index('note_inc' if note_diff > 0 else 'note_dec')] * abs(note_diff)

        if event_type == 'note_on':
            # velocity = (msg.velocity) // (MIDI_VELOCITY // VEL_QUANTIZATION)
            tokens.append(TOKEN_IDS.index('note_on'))
        elif event_type == 'note_off':
            # velocity = 0
            tokens.append(TOKEN_IDS.index('note_off'))

        # If velocity is different, we update it
        # if last_velocity != velocity:
            # TODO: Reimplement velocity
            # events.append(VEL_OFFSET + velocity)
            # last_velocity = velocity

    return np.array(tokens, dtype='B')

def tokens_to_str(tokens):
    return ''.join(map(str, tokens))

def str_to_tokens(string):
    return np.array(list(map(int, string.strip())), dtype='B')

def load_midi(fname, no_cache=False):
    cache_path = os.path.join(CACHE_DIR, fname + '.npy')
    try:
        if no_cache:
            raise Exception()
        seq = np.load(cache_path)
    except Exception as e:
        # Load
        mid = mido.MidiFile(fname)
        track = mido.merge_tracks(mid.tracks)
        seq = midi_to_tokens(mid, track)

        # Perform caching
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, seq)

    return seq

def synthesize(mid_fname, gain=3.3):
    """
    Synthesizes a MIDI file into MP3
    """
    # Find soundfont
    if not os.path.isfile(SOUND_FONT_PATH):
        # Download it
        urllib.request.urlretrieve(SOUND_FONT_URL, SOUND_FONT_PATH)

    # Synthsize
    fsynth_proc = subprocess.Popen([
        'fluidsynth',
        '-nl',
        '-f', 'fluidsynth.cfg',
        '-T', 'raw',
        '-g', str(gain),
        '-F', '-',
        SOUND_FONT_PATH,
        mid_fname
    ], stdout=subprocess.PIPE)

    # Convert to MP3
    lame_proc = subprocess.Popen(['lame', '-q', '5', '-r', '-'], stdin=fsynth_proc.stdout, stdout=subprocess.PIPE)
    return lame_proc.communicate()[0]

if __name__ == '__main__':
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load("out/token.model")
    # Encode
    tokens = load_midi('data/classical/Beethoven/Sonata Op 53 1st mvmt.mid', no_cache=True)
    token_str = tokens_to_str(tokens)
    print('Token string len', len(token_str))
    token_ids = sp.EncodeAsIds(token_str)
    print('Token ID Length', len(token_ids))
    print(token_ids[:20])

    # Decode
    d_token_str = sp.DecodeIds(token_ids)
    assert token_str == d_token_str
    d_tokens = str_to_tokens(d_token_str)
    assert (tokens == d_tokens).all()
    midi = tokens_to_midi(d_tokens)
    midi.save('out/en_dec.mid')