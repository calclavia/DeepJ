from preprocess import midi_io
from preprocess import melodies_lib

# Test midi load and save
seq_pb = midi_io.midi_to_sequence_proto('data/edm/edm_c/rc_3.mid')
midi_io.sequence_proto_to_midi(seq_pb).write('out/rw_pb.mid')

# Extract melody
melody = melodies_lib.midi_file_to_melody(seq_pb)
mel_pb = melody.to_sequence()
midi_io.sequence_proto_to_midi(mel_pb).write('out/melody_pb.mid')
