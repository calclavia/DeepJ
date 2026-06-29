#!/usr/bin/env bash
# Runnable gate for DeepJ portability (no trained weights required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

"$PY" -m unittest test_portability -v
PYTHONPATH="$ROOT" "$PY" scripts/list_styles.py | grep -q bach
if "$PY" -c "import midi" 2>/dev/null; then
  "$PY" -m unittest test.TestMIDIUtil.test_encode test.TestMIDIUtil.test_decode -v
else
  echo "skip: python-midi not installed (midi encode/decode tests)"
fi
echo "GATE GREEN: deepj portability"