"""Build task-order manifests for the ALFWorld stream.

ALFWorld plays its train games in a fixed order (the environment shuffles the
sorted game list once with seed 1234 and then walks it). The default stream is
the first N games of that order; a manifest is an explicit permutation of those
same N games, used to test robustness to task order.

    export ALFWORLD_DATA=...
    python data/build_order_manifests.py --n 600 --seeds 101 202 303 404 --out data/manifests/alf_order
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def canonical_order(alfworld_data: str, n: int) -> list[str]:
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    from memcryst.agent.alfworld_env import AlfworldEnv, _ensure_default_config

    os.environ["ALFWORLD_DATA"] = alfworld_data
    cfg = AlfworldEnv._load_config(_ensure_default_config())
    raw = AlfredTWEnv(cfg, train_eval="train")
    files = sorted(raw.game_files)
    np.random.RandomState(1234).shuffle(files)
    return [os.path.relpath(p, alfworld_data) for p in files[:n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404])
    ap.add_argument("--out", default="data/manifests/alf_order")
    args = ap.parse_args()
    data = os.environ.get("ALFWORLD_DATA")
    if not data:
        raise SystemExit("export ALFWORLD_DATA first (see data/prepare_alfworld.sh)")
    canon = canonical_order(data, args.n)
    sha = hashlib.sha256("\n".join(canon).encode()).hexdigest()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"canonical": "sorted train game list shuffled once with numpy RandomState(1234); first n",
            "n": args.n, "canonical_sha256": sha}
    json.dump({**meta, "order_seed": None, "gamefiles": canon}, open(out / f"canonical_first{args.n}.json", "w"), indent=1)
    for seed in args.seeds:
        perm = np.random.RandomState(seed).permutation(args.n)
        json.dump({**meta, "order_seed": seed,
                   "permutation_rule": f"numpy RandomState(order_seed).permutation({args.n}) applied to the canonical list",
                   "canonical_position": [int(k) for k in perm],
                   "gamefiles": [canon[int(k)] for k in perm]},
                  open(out / f"order_seed{seed}.json", "w"), indent=1)
    print(f"wrote {len(args.seeds) + 1} manifests to {out} (canonical sha256 {sha[:16]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
