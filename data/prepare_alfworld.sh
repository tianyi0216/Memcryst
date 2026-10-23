#!/bin/bash
# prepare_alfworld.sh — download the ALFWorld text games and point ALFWORLD_DATA at them.
# Usage: bash data/prepare_alfworld.sh [target_dir]      (default: data/raw/alfworld)
set -e
TARGET=${1:-data/raw/alfworld}
mkdir -p "$TARGET"
python -c "import alfworld" 2>/dev/null || pip install alfworld
export ALFWORLD_DATA="$(cd "$TARGET" && pwd)"
alfworld-download
echo "ALFWorld data in $ALFWORLD_DATA ($(find "$ALFWORLD_DATA/json_2.1.1/train" -name 'game.tw-pddl' | wc -l) train games)"
echo "export ALFWORLD_DATA=$ALFWORLD_DATA   # or set env.alfworld_data in the config"
