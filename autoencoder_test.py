import torch
import torch.nn as nn
import torch.nn.functional as F
from model import EncoderRNN, DecoderRNN, AutoEncoder
from midi_io import *
from constants import *

# HIDDEN_SIZE = 512
model = AutoEncoder()
model.load_state_dict(torch.load('out/model_VAE.pt'))

note_seq = load_midi('data/baroque/Bach/Duet in E Minor BWV802.mid')
note_seq = torch.from_numpy(note_seq).unsqueeze(0)
note_seq = var(note_seq, volatile=True).cpu()
x = model.embd(note_seq)
encoder_output, encoder_hidden = model.encoder(x, None)
# test_path = os.path.join('out', 'test_latent_vector.npy')
# np.save(test_path, encoder_hidden)
decoder_output, decoder_hidden = model.decoder(x, encoder_hidden)
decoder_output = F.softmax(decoder_output, dim=2)
output_max = torch.max(decoder_output, 2)
save_midi('autoencoder_test', output_max[1].squeeze(0).data.cpu().numpy())
