#!/usr/bin/env python
# encoding: utf-8

"""
Copyright (C) 2012 Devon Bryant
"""
import os, sys, getopt, glob, random, re, subprocess

def is_fsynth_installed():
    """ Check to make sure fluidsynth exists in the PATH """
    for path in os.environ['PATH'].split(os.pathsep):
        f = os.path.join(path, 'fluidsynth')
        if os.path.exists(f) and os.access(f, os.X_OK):
            return True

    return False

def to_audio(sf2, midi_file, out_dir, out_type='wav', txt_file=None, append=True):
    """
    Convert a single midi file to an audio file.  If a text file is specified,
    the first line of text in the file will be used in the name of the output
    audio file.  For example, with a MIDI file named '01.mid' and a text file
    with 'A    major', the output audio file would be 'A_major_01.wav'.  If
    append is false, the output name will just use the text (e.g. 'A_major.wav')

    Args:
        sf2 (str):        the file path for a .sf2 soundfont file
        midi_file (str):  the file path for the .mid midi file to convert
        out_dir (str):    the directory path for where to write the audio out
        out_type (str):   the output audio type (see 'fluidsynth -T help' for options)
        txt_file (str):   optional text file with additional information of how to name
                          the output file
        append (bool):    whether or not to append the optional text to the original
                          .mid file name or replace it
    """
    fbase = os.path.splitext(os.path.basename(midi_file))[0]
    if not txt_file:
        out_file = out_dir + '/' + fbase + '.' + out_type
    else:
        line = 'out'
        with open(txt_file, 'r') as f:
            line = re.sub(r'\s', '_', f.readline().strip())

        if append:
            out_file = out_dir + '/' + line + '_' + fbase + '.' + out_type
        else:
            out_file = out_dir + '/' + line + '.' + out_type

    subprocess.call(['fluidsynth', '-T', out_type, '-F', out_file, '-ni', sf2, midi_file])

def main():
    """
    Convert a directory of MIDI files to audio files using the following command line options:

    --sf2-dir (required)   the path to a directory with .sf2 soundfont files.  The script will
                           pick a random soundfont from this directory for each file.

    --midi-dir (required)  the path to a directory with the .mid MIDI files to convert.

    --out-dir (optional)   the directory to write the audio files to

    --type (optional)      the audio type to write out (see 'fluidsynth -T help' for options)
                           the default is 'wav'

    --replace (optional)   if .txt files exist in the same directory as the .mid files, the text
                           from the files will be used for the output audio file names instead
                           of the midi file names.  If not specified, the text from the files will
                           be appended to the file name.
    """
    try:
        if not is_fsynth_installed():
            raise Exception('Unable to find \'fluidsynth\' in the path')

        opts, args = getopt.getopt(sys.argv[1:], None, ['sf2-dir=', 'midi-dir=', 'out-dir=', 'type=', 'replace'])
        sf2files, midifiles, textfiles, out_dir, out_type, append = [], [], [], None, 'wav', True
        for o, v in opts:
            if o == '--sf2-dir':
                sf2files = glob.glob(v + '/*.[sS][fF]2')
            elif o == '--midi-dir':
                midifiles = glob.glob(v + '/*.[mM][iI][dD]')
                textfiles = glob.glob(v + '/*.[tT][xX][tT]')
                if not out_dir:
                    out_dir = v
            elif o == '--out-dir':
                out_dir = v
            elif o == '--type':
                out_type = v
            elif o == '--replace':
                append = False

        if not sf2files:
            raise Exception('A --sf2-dir directory must be specified where at least one .sf2 file exists')
        elif not midifiles:
            raise Exception('A --midi-dir directory must be specified where at least one .mid file exists')

        if not textfiles or len(textfiles) < len(midifiles):
            for mid in midifiles:
                to_audio(random.choice(sf2files), mid, out_dir, out_type)
        else:
            for mid, txt in zip(midifiles, textfiles):
                to_audio(random.choice(sf2files), mid, out_dir, out_type, txt, append)

    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)
    except Exception as exc:
        print(str(exc))
        sys.exit(2)

if __name__ == '__main__':
    sys.exit(main())
