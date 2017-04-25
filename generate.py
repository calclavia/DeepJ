import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import trange

from midi_util import *
from dataset import unclamp_midi
from constants import *
from model import DeepJ

def generate(model, name='output', num_bars=16):
    model.eval()

    note_seq = []

    # RNN state
    states = None

    # Last generated note time step
    prev_note = Variable(torch.zeros(NUM_NOTES), volatile=True).cuda().unsqueeze(0)

    for i in trange(NOTES_PER_BAR * num_bars):
        ## Time Axis
        note_features, states = model.time_axis(prev_note, states)

        ## Note Axis
        # The current note being generated
        current_note = Variable(torch.zeros(NUM_NOTES), volatile=True).cuda().unsqueeze(0)

        for n in range(NUM_NOTES):
            prob = model.note_axis(note_features, current_note)
            prob = prob.cpu().data

            # Sample note randomly
            current_note[0, n] = 1 if np.random.random() <= prob[n, 0] else 0

        prev_note = current_note
        # Add note to note sequence
        note_seq.append(current_note.cpu().data[0, :].numpy())

    note_seq = np.array(note_seq)
    # TODO: Implement articulation
    replay_seq = np.zeros_like(note_seq)
    write_file(name, note_seq, replay_seq)

def write_file(name, note_seq, replay_seq):
    """
    Takes a list of all notes generated per track and writes it to file
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    fpath = SAMPLES_DIR + '/' + name + '.mid'
    print('Writing file', fpath)
    mf = midi_encode(unclamp_midi(note_seq), unclamp_midi(replay_seq))
    midi.write_midifile(fpath, mf)

def main():
    print('=== Loading Model ===')
    model = torch.load(OUT_DIR + '/model.pt')

    print('=== Generating ===')
    print('GPU: {}'.format(torch.cuda.is_available()))
    generate(model)

if __name__ == '__main__':
    main()
