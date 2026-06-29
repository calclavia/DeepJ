import os
import numpy as np
import tensorflow as tf
from collections import deque
import midi
import argparse

from constants import *
from util import *
from dataset import *
from tqdm import tqdm
from midi_util import midi_encode


class MusicGeneration:
    """Represents a music generation run."""

    def __init__(self, style, default_temp=1):
        self.notes_memory = deque([np.zeros((NUM_NOTES, NOTE_UNITS)) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        self.beat_memory = deque([np.zeros(NOTES_PER_BAR) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        self.style_memory = deque([style for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        self.silent_time = NOTES_PER_BAR
        self.results = []
        self.default_temp = default_temp
        self.temperature = default_temp

    def build_time_inputs(self):
        return (
            np.array(self.notes_memory),
            np.array(self.beat_memory),
            np.array(self.style_memory),
        )

    def build_note_inputs(self, note_features):
        return (
            np.array(note_features),
            np.array([self.next_note]),
            np.array(list(self.style_memory)[-1:]),
        )

    def choose(self, prob, n):
        vol = prob[n, -1]
        prob = apply_temperature(prob[n, :-1], self.temperature)
        if np.random.random() <= prob[0]:
            self.next_note[n, 0] = 1
            self.next_note[n, 2] = vol
            if np.random.random() <= prob[1]:
                self.next_note[n, 1] = 1

    def end_time(self, t):
        if np.count_nonzero(self.next_note) == 0:
            self.silent_time += 1
            if self.silent_time >= NOTES_PER_BAR:
                self.temperature += 0.1
        else:
            self.silent_time = 0
            self.temperature = self.default_temp

        self.notes_memory.append(self.next_note)
        self.beat_memory.append(compute_beat(t, NOTES_PER_BAR))
        self.results.append(self.next_note)
        self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        return self.results[-1]


def apply_temperature(prob, temperature):
    if temperature != 1:
        x = -np.log(1 / prob - 1)
        prob = 1 / (1 + np.exp(-x / temperature))
    return prob


def process_inputs(ins):
    ins = list(zip(*ins))
    return [np.array(i) for i in ins]


def generate(models, num_bars, styles):
    print('Generating with styles:', styles)
    _, time_model, note_model = models
    generations = [MusicGeneration(style) for style in styles]

    for t in tqdm(range(NOTES_PER_BAR * num_bars)):
        ins = process_inputs([g.build_time_inputs() for g in generations])
        note_features = time_model.predict(ins)
        note_features = np.array(note_features)[:, -1:, :]

        for n in range(NUM_NOTES):
            ins = process_inputs([
                g.build_note_inputs(note_features[i, :, :, :])
                for i, g in enumerate(generations)
            ])
            predictions = np.array(note_model.predict(ins))
            for i, g in enumerate(generations):
                g.choose(predictions[i][-1], n)

        yield [g.end_time(t) for g in generations]


def write_file(name, results):
    results = zip(*list(results))
    for i, result in enumerate(results):
        fpath = os.path.join(SAMPLES_DIR, name + '_' + str(i) + '.mid')
        print('Writing file', fpath)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        mf = midi_encode(unclamp_midi(result))
        midi.write_midifile(fpath, mf)


def main():
    parser = argparse.ArgumentParser(
        description='Generate polyphonic MIDI with composer-style conditioning (DeepJ).',
        epilog='Paper: arxiv:1801.00887. List indices: python scripts/list_styles.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--bars', default=32, type=int, help='Number of bars to generate')
    parser.add_argument(
        '--styles', default=None, type=int, nargs='+',
        help='Composer style indices to mix (uniform average)',
    )
    parser.add_argument(
        '--genre', default=None, type=int, choices=range(len(genre)),
        help='One genre track (0=baroque, 1=classical, 2=romantic)',
    )
    parser.add_argument('--output', default='output', help='Output filename prefix under out/samples/')
    args = parser.parse_args()

    models = build_or_load()

    if args.styles:
        styles = [np.mean([one_hot(i, NUM_STYLES) for i in args.styles], axis=0)]
    elif args.genre is not None:
        styles = [compute_genre(args.genre)]
    else:
        styles = [compute_genre(i) for i in range(len(genre))]

    write_file(args.output, generate(models, args.bars, styles))


if __name__ == '__main__':
    main()