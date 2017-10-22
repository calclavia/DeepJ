import os
import sys
import logging

from flask import stream_with_context, request, Response
from flask import Flask

import torch
from model import DeepJ

from uuid import uuid4
from midi_io import *
from subprocess import call
from generate import Generation

# Global log config
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

path = os.path.dirname(__file__)

# Load model
model = DeepJ()
# Load tensors onto the CPU
saved_obj = torch.load(os.path.join(path, 'archives/model.pt'), map_location=lambda storage, loc: storage)
model.load_state_dict(saved_obj)

# Synth parameters
soundfont = os.path.join(path, 'acoustic_grand_piano.sf2')
gain = 1

# Warm up Pytorch
print('Warming up Pytorch')
Generation(model).generate(seq_len=10, show_progress=False) 

@app.route('/stream.wav')
def streamed_response():
    def generate():
        # style = request.args['style']
        uuid = uuid4()
        logger.info('Stream ID: {}'.format(uuid))
        folder = os.path.join('/tmp', str(uuid))

        os.makedirs(folder, exist_ok=True)

        mid_fname = os.path.join(folder, 'generation.mid')
        wav_fname = os.path.join(folder, 'generation.wav')

        logger.info('Generating MIDI')
        seq = Generation(model).generate(seq_len=5000, show_progress=False) 
        midi_file = seq_to_midi(seq)
        midi_file.save(mid_fname)

        logger.info('Synthesizing MIDI')
        call(['fluidsynth', '-F', wav_fname, '-g', str(gain), soundfont, mid_fname])

        logger.info('Streaming data')

        with open(wav_fname, "rb") as f:
            data = f.read(1024)
            while data:
                yield data
                data = f.read(1024)
                
        # Clean up the temporary files
        os.remove(mid_fname)
        os.remove(wav_fname)
        os.rmdir(folder)
    return Response(stream_with_context(generate()), mimetype='audio/wav')

@app.route('/')
def index():
    return """
    Streaming Audio
    <audio autoplay loop><source src="/stream.wav" type="audio/wav"></audio>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0')