from preprocess import midi_io

# Test data loading
pb = midi_io.midi_to_sequence_proto('data/classical_c/mz_330_1.mid')
