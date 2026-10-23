#!/bin/bash
# run_dialsim.sh — lifelong run on the conversational QA stream.
# Usage: bash run_dialsim.sh [config.yaml]       (default: configs/dialsim_full.yaml)
set -e
CONFIG=${1:-configs/dialsim_full.yaml}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -u src/run.py --config "$CONFIG"
