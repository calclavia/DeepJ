import matplotlib.pyplot as plt
import numpy as np
import dataset
from music import NUM_CLASSES, MIN_CLASS, NOTES_PER_BEAT, NOTE_OFF, NO_EVENT, MIN_NOTE

MIDI_NOTE_RANGE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                    'C']

def plot_note_distribution(paths):
    melody_list = dataset.load_melodies(paths)
    for i, melody in enumerate(melody_list):
        # Filter out 0's and 1's
        notes = list(filter(lambda x: x != 0 and x != 1, melody))
        # Subtract min class from each note to 0 index the whole list
        notes = [x - MIN_CLASS for x in notes]
        #print('new melody list: ', notes)
        #print('note count: ', np.bincount(notes))
        plt.hist(notes, align='mid', rwidth=0.5)
        plt.xticks(range(49), MIDI_NOTE_RANGE)
        #plt.show()
        plt.savefig('note_dist_' + str(i) + '.png')

plot_note_distribution(['test'])

"""
NOTES:
2 maps to midi note 36 (MIN_NOTE)
8 numbers in arr forms a bar
2 elements in arr are quarter note
1 element is a half a quarter note
output a png of plot with plotpy save
"""
