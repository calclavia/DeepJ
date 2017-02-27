import matplotlib.pyplot as plt
import numpy as np
import sys
import dataset
from music import NUM_CLASSES, MIN_CLASS, NOTES_PER_BEAT, NOTE_OFF, NO_EVENT, MIN_NOTE

MIDI_NOTE_RANGE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                    'C']
NOTE_LEN_RANGE = ['0', '',
                    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                    'C']

def plot_note_distribution(paths):
    melody_list = dataset.load_melodies(paths)
    for i, melody in enumerate(melody_list):
        fig = plt.figure(figsize=(12, 5))
        # Filter out 0's and 1's
        notes = list(filter(lambda x: x != 0 and x != 1, melody))
        # Subtract min class from each note to 0 index the whole list
        notes = [x - MIN_CLASS for x in notes]
        # Plot
        plt.hist(notes, bins=np.arange(50)-0.5, rwidth=0.8)
        plt.ylabel('Number of note occurances')
        plt.xticks(range(len(MIDI_NOTE_RANGE)), MIDI_NOTE_RANGE)
        # plt.show()
        plt.savefig('note_dist_' + str(i) + '.png')

def plot_note_length(paths):
    melody_list = dataset.load_melodies(paths)
    for i, melody in enumerate(melody_list):
        # Dict that stores notes and their lengths
        note_len_dict = {}
        # Initialize keys/values in dict
        for i in range(51):
            note_len_dict[i] = 0

        prev_note = 0
        for m in melody:
            # Note off
            if m == 0:
                note_len_dict[0] += 1
            # No event
            elif m == 1:
                note_len_dict[prev_note] += 1
            # Note
            else:
                note_len_dict[m] += 1
                prev_note = m
        # Convert dict into a list that can be put into histogram
        note_lens = []
        for k in note_len_dict.keys():
            for i in range(note_len_dict[k]):
                note_lens.append(k)

        # Plot
        fig = plt.figure(figsize=(12, 5))
        plt.hist(note_lens, bins=np.arange(52)-0.5, rwidth=0.8)
        plt.xlabel('0 represents a rest')
        plt.ylabel('Duration in eigth notes')
        plt.xticks(range(len(NOTE_LEN_RANGE)), NOTE_LEN_RANGE)
        # plt.show()
        plt.savefig('note_len_dist_' + str(i) + '.png')

def distributions(paths):
    plot_note_distribution(paths)
    plot_note_length(paths)

distributions(sys.argv)

"""
NOTES:
2 maps to midi note 36 (MIN_NOTE)
8 numbers in arr forms a bar
2 elements in arr are quarter note
1 element is a half a quarter note, or an eigth note
output a png of plot with plotpy save
"""
