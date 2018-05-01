import os
import argparse
import numpy as np
from numpy import linalg as LA
from midi_io import load_midi

def main():
    parser = argparse.ArgumentParser(description='Apply Spherical Linear Interpolation between two pieces of music')
    parser.add_argument('--song-a', help='Path to first piece of music')
    parser.add_argument('--song-b', help='Path to second piece of music')
    parser.add_argument('--mu', default=0, type=float, help='Scalar representing how much to interpolate between song A and B')
    args = parser.parse_args()
    mu = args.mu
    
    try:
        # Convert songs to lists of midi events
        seq_a = load_midi(args.song_a)
        seq_b = load_midi(args.song_b)

        # Pad smaller song with zeros
        diff = len(seq_a) - len(seq_b)
        if diff > 0:
            seq_b = np.pad(seq_b, (0, diff), 'constant')
        if diff < 0:
            seq_a = np.pad(seq_a, (0, abs(diff)), 'constant')
        
        # Apply slerp
        seq_a = seq_a / LA.norm(seq_a)
        seq_b = seq_b / LA.norm(seq_b)
        theta = np.arccos(np.dot(seq_a, seq_b))
        slerp = (np.sin(1 - mu) * theta / np.sin(theta)) * seq_a + (np.sin(mu) * theta / np.sin(theta)) * seq_b
        print('slerp: ', slerp)
    except Exception as e:
        print('Error: ', e)

if __name__ == '__main__':
    main()
