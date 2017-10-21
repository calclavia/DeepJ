import time
import numpy
import fluidsynth

import wave
from wave import Wave_write
from tempfile import SpooledTemporaryFile

import os

path = os.path.dirname(__file__)
soundfont = os.path.join(path, 'acoustic_grand_piano.sf2')

def synth():
    s = []

    fl = fluidsynth.Synth()

    # Initial silence is 1 second
    s = numpy.append(s, fl.get_samples(44100 * 1))

    sfid = fl.sfload(soundfont)
    fl.program_select(0, sfid, 0, 0)

    fl.noteon(0, 60, 30)
    fl.noteon(0, 67, 30)
    fl.noteon(0, 76, 30)

    # Chord is held for 2 seconds
    s = numpy.append(s, fl.get_samples(44100 * 2))

    fl.noteoff(0, 60)
    fl.noteoff(0, 67)
    fl.noteoff(0, 76)

    # Decay of chord is held for 1 second
    s = numpy.append(s, fl.get_samples(44100 * 1))

    fl.delete()

    samps = fluidsynth.raw_audio_string(s)

    file = SpooledTemporaryFile()
    with wave.open(file, 'rb') as f:
        f.writeframesraw(samps)
    return file