#!/bin/bash
# run_ablations.sh — the ALFWorld layer ablations, one full run each (sequential).
# Usage: bash run_ablations.sh
set -e
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
for cfg in configs/alfworld_vanilla.yaml configs/alfworld_memory_only.yaml \
           configs/alfworld_skill_only.yaml configs/alfworld_no_skill_revision.yaml; do
    echo "=== $cfg ==="
    python -u src/run.py --config "$cfg"
done
python src/metrics.py runs/alfworld_vanilla/lifelong.json runs/alfworld_memory_only/lifelong.json \
       runs/alfworld_skill_only/lifelong.json runs/alfworld_no_skill_revision/lifelong.json
