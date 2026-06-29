#!/usr/bin/env bash
# Setup DeepJ deps (python-midi + pip). Idempotent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
MIDI_SRC="${PYTHON_MIDI_SRC:-$HOME/src/python-midi}"

if ! "$PY" -c "import midi" 2>/dev/null; then
  if [ ! -d "$MIDI_SRC" ]; then
    echo "clone python-midi to $MIDI_SRC (or set PYTHON_MIDI_SRC)" >&2
    mkdir -p "$(dirname "$MIDI_SRC")"
    git clone --depth 1 https://github.com/vishnubob/python-midi.git "$MIDI_SRC"
  fi
  (cd "$MIDI_SRC" && "$PY" setup.py install --user)
fi

REQ="${DEEPJ_REQUIREMENTS:-requirements-modern.txt}"
"$PY" -m pip install --user -r "$REQ"
echo "DeepJ setup OK ($REQ)"