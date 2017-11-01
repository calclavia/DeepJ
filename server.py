import os
import sys
import logging
import numpy as np
import subprocess
from functools import partial
from tempfile import TemporaryFile

from flask import Flask, stream_with_context, request, Response, render_template

import torch
from model import DeepJ

import mido
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
gain = 4.3

styles = {
    'baroque': 0,
    'classical': 1,
    'romantic': 2,
    'modern': 3
}

@app.route('/stream.mp3')
def streamed_response():
    # Determine style
    gen_style = []

    for style, style_id in styles.items():
        strength = request.args.get(style, 0)
        gen_style.append(one_hot(style_id, NUM_STYLES) * float(strength))

    gen_style = np.mean(gen_style, axis=0)

    if np.sum(gen_style) > 0:
        # Normalize
        gen_style /= np.sum(gen_style)
    else:
        gen_style = None

    seq_len = max(min(int(request.args.get('length', 500)), 100000), 0)

    uuid = uuid4()
    logger.info('Stream ID: {}'.format(uuid))
    logger.info('Style: {}'.format(gen_style))
    folder = '/tmp'
    mid_fname = os.path.join(folder, '{}.mid'.format(uuid))

    logger.info('Generating MIDI')
    seq = Generation(model, style=gen_style, default_temp=0.93).generate(seq_len=seq_len, show_progress=False)         
    track_builder = TrackBuilder(iter(seq), tempo=mido.bpm2tempo(90))
    track_builder.run()
    midi_file = track_builder.export()
    midi_file.save(mid_fname)

    logger.info('Synthesizing MIDI')

    # Synthsize
    fsynth_proc = subprocess.Popen([
        'fluidsynth',
        '-nl',
        '-f', 'fluidsynth.cfg',
        '-T', 'raw',
        '-g', str(gain),
        '-F', '-',
        soundfont,
        mid_fname
    ], stdout=subprocess.PIPE)

    # Convert to MP3
    lame_proc = subprocess.Popen(['lame', '-q', '2', '-r', '-'], stdin=fsynth_proc.stdout, stdout=subprocess.PIPE)

    logger.info('Streaming data')
    data, err = lame_proc.communicate()
    os.remove(mid_fname)
    return Response(data, mimetype='audio/mp3')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
