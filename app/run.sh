#!/bin/bash
# VideoPolish-er wrapper script
# This script uses the Transcribix venv for dependencies

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSCRIBIX_VENV="${TRANSCRIBIX_VENV:-$HOME/Code/Transcribix/venv/bin/python}"

if [ ! -f "$TRANSCRIBIX_VENV" ]; then
    echo "Error: Transcribix venv not found at $TRANSCRIBIX_VENV"
    echo "Please install Transcribix first or update the path in this script."
    exit 1
fi

exec "$TRANSCRIBIX_VENV" "$SCRIPT_DIR/cli.py" "$@"