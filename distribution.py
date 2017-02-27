import matplotlib.pyplot as plt
import numpy as np
import sys
import dataset
import ntpath

from music import autocorrelate, NUM_CLASSES, MIN_CLASS, NOTES_PER_BEAT, NOTE_OFF, NO_EVENT, MIN_NOTE

MIDI_NOTE_RANGE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] * 4 + ['C']
NOTE_LEN_RANGE = ['0', ''] + MIDI_NOTE_RANGE

def plot_note_distribution(melody_list):
    for i, (name, melody) in enumerate(melody_list):
        fig = plt.figure(figsize=(14, 5))
        # Filter out 0's and 1's
        # Subtract min class from each note to 0 index the whole list
        notes = [x - MIN_CLASS for x in melody if x != 0 and x != 1]
        # Plot
        plt.hist(notes, bins=np.arange(len(MIDI_NOTE_RANGE) + 1))
        plt.ylabel('Note frequency')
        plt.xticks(range(len(MIDI_NOTE_RANGE)), MIDI_NOTE_RANGE)
        # plt.show()
        plt.savefig('out/' + ntpath.basename(name) + ' (note dist).png')

def plot_note_length(melody_list):
    for i, (name, melody) in enumerate(melody_list):
        # Dict that stores notes and their lengths
        note_len_dict = {}
        # Initialize keys/values in dict
        for i in range(len(NOTE_LEN_RANGE)):
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
        fig = plt.figure(figsize=(14, 5))
        plt.hist(note_lens, bins=np.arange(len(NOTE_LEN_RANGE) + 1))
        plt.xlabel('0 represents a rest')
        plt.ylabel('Duration in eigth notes')
        plt.xticks(range(len(NOTE_LEN_RANGE)), NOTE_LEN_RANGE)
        # plt.show()
        plt.savefig('out/' + ntpath.basename(name) + ' (note length).png')

def calculate_correlation(melody_list):
    correlations = []
    for name, melody in melody_list:
        correlation =  np.sum([autocorrelate(melody, i) ** 2 for i in range(1, 4)])
        correlations.append(correlation)
        print('Correlation Coefficient (r^2 for 1, 2, 3): ', name, correlation)

    print('Mean: ', np.mean(correlations))
    print('Std: ', np.std(correlations))

def distributions(paths):
    melody_list = dataset.load_melodies(paths, shuffle=False, named=True)
    plot_note_distribution(melody_list)
    plot_note_length(melody_list)
    calculate_correlation(melody_list)

distributions(sys.argv)

"""
NOTES:
2 maps to midi note 36 (MIN_NOTE)
8 numbers in arr forms a bar
2 elements in arr are quarter note
1 element is a half a quarter note, or an eigth note
output a png of plot with plotpy save
"""
