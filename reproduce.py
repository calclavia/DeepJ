import torch
import torch.nn as nn
import torch.nn.functional as F
from model import EncoderRNN, DecoderRNN, DeepJ
from midi_io import *
from constants import *

# HIDDEN_SIZE = 512
model = DeepJ().cuda()
model.load_state_dict(torch.load('out/model_VAE.pt'))

x = load_midi('data/baroque/Bach/Duet in E Minor BWV802.mid')
x = torch.from_numpy(x).unsqueeze(0)
# Slice based on the same seq len it's trained on
# x = x[:, :SEQ_LEN]
x = var(x, volatile=True)

x = model.embd(x)
mean, logvar, _ = model.encoder(x, None)
z = mean
# This will be a deterministic encoding?

decoder_output, _ = model.decoder(x[:, :-1], z)
decoder_output = F.softmax(decoder_output, dim=2)
output_max = torch.max(decoder_output, 2)
save_midi('autoencoder_test', output_max[1].squeeze(0).data.cpu().numpy())
