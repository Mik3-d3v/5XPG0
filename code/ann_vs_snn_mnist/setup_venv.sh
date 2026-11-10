#!/bin/bash
# setup_venv.sh — create a local virtual environment and install the training
# dependencies. Run it once, then activate it (the script prints the command).
set -euo pipefail
cd "$(dirname "$0")"

# python3 on Linux/macOS, python on Windows (Git Bash)
PY=$(command -v python3 || command -v python)
# only create the venv once; re-creating it fails on Windows while its
# python.exe is in use (e.g. by VS Code or an activated terminal)
if [ ! -f .venv/pyvenv.cfg ]; then "$PY" -m venv .venv; fi

# venvs use bin/ on Linux/macOS but Scripts/ on Windows
if [ -f .venv/bin/activate ]; then ACTIVATE=.venv/bin/activate; else ACTIVATE=.venv/Scripts/activate; fi
source "$ACTIVATE"
python -m pip install --upgrade pip
pip install -r requirements.txt
echo
echo "done. activate with:  source $ACTIVATE"
