import os
import numpy as np
import midi
import torch

from constants import *
from util import *
from generate import *
from model import DeepJ

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, send, emit, disconnect
from threading import Thread

app = Flask(__name__)
async_mode = None
socketio = SocketIO(app, async_mode=async_mode)

# Load model 
model = DeepJ()

if torch.cuda.is_available():
    model.cuda()

model_path = 'archives/model.pt'

if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))
else:
    print('WARNING: No model loaded! Please make sure a model exists.')

# TODO: Add style

thread = None

@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)

@socketio.on('client_connect')
def handle_client_connect(json):
    print('Received JSON: {0}'.format(str(json)))
    global thread
    if thread is None:
        thread = socketio.start_background_task(generate_infinite)

@socketio.on('client_disconnect')
def handle_client_disconnect():
    # Stop and delete thread generating timesteps
    print('Client disconnected.')
    disconnect()

# @socketio.on('play_event')
# def handle_play_event(json):
#     print('Received JSON: ' + str(json))
#     timestep = generate()
#     emit('send_timestep', timestep)

def generate_infinite():
    prev_timestep = var(torch.zeros((NUM_NOTES, NOTE_UNITS)), volatile=True).unsqueeze(0)
    states = None
    for t in range(1000000):
        beat = var(to_torch(compute_beat(t, NOTES_PER_BAR)), volatile=True).unsqueeze(0)
        curr_timestep, s = model.generate(prev_timestep, beat, states)
        # TODO: Add temperature
        prev_timestep = curr_timestep
        states = s
        # Transform variable into list. CUDA tensor doesn't support GPU array so cpu() is required 
        timestep = curr_timestep.data.cpu().numpy().tolist()[0]
        print('TIMESTEP: ', timestep)
        socketio.emit('send_timestep', timestep)

if __name__ == "__main__":
    print('Running socket io')
    socketio.run(app)
