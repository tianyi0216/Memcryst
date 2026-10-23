#!/bin/bash
# eval.sh — summarize one or more finished runs (success rate, per family, per cycle).
# Usage: bash eval.sh runs/<name>/lifelong.json [more.json ...]
set -e
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python src/metrics.py "$@"
