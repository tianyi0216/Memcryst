#!/bin/bash
# run_alfworld.sh — lifelong wake-sleep run on ALFWorld.
# Usage: bash run_alfworld.sh [config.yaml]      (default: configs/alfworld_full.yaml)
set -e
CONFIG=${1:-configs/alfworld_full.yaml}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -u src/run.py --config "$CONFIG"
