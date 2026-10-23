#!/bin/bash
# run_order_robustness.sh — the full recipe under four shuffled task orders.
# Usage: bash run_order_robustness.sh
set -e
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
for seed in 101 202 303 404; do
    echo "=== order seed $seed ==="
    python -u src/run.py --config configs/alfworld_full.yaml \
        --set run.name=alfworld_order_$seed \
        --set env.order_manifest=data/manifests/alf_order/order_seed$seed.json
done
python src/metrics.py runs/alfworld_order_101/lifelong.json runs/alfworld_order_202/lifelong.json \
       runs/alfworld_order_303/lifelong.json runs/alfworld_order_404/lifelong.json
