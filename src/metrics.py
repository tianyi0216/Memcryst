"""Summarize lifelong runs: prequential success rate, per-family rates, per-cycle
curve, and the first/last-quarter change.

    python src/metrics.py runs/<name>/lifelong.json [more.json ...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter


def load(path: str) -> list[dict]:
    d = json.load(open(path))
    return [t for c in d.get("cycles", []) for t in c["wake"]["per_task"]]


def summarize(path: str) -> dict:
    per = load(path)
    n = len(per)
    succ = sum(1 for t in per if t.get("success"))
    fam = Counter(t.get("family") for t in per)
    fam_s = Counter(t.get("family") for t in per if t.get("success"))
    q = max(n // 4, 1)
    d = json.load(open(path))
    cycles = [sum(1 for t in c["wake"]["per_task"] if t.get("success")) / max(len(c["wake"]["per_task"]), 1)
              for c in d.get("cycles", [])]
    return {
        "path": path, "n": n, "success": succ, "sr": 100 * succ / max(n, 1),
        "family": {f: (fam_s[f], fam[f]) for f in sorted(fam)},
        "first_quarter": 100 * sum(1 for t in per[:q] if t.get("success")) / q,
        "last_quarter": 100 * sum(1 for t in per[-q:] if t.get("success")) / q,
        "per_cycle": [round(100 * c, 1) for c in cycles],
        "mean_steps": sum(t.get("steps", 0) for t in per) / max(n, 1),
    }


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    rows = [summarize(p) for p in argv]
    for r in rows:
        print(f"\n=== {r['path']} ===")
        print(f"  success rate: {r['sr']:.2f}%  ({r['success']}/{r['n']})   mean steps {r['mean_steps']:.1f}")
        print("  per family:   " + "  ".join(f"{f} {s}/{c} ({100*s/max(c,1):.1f}%)" for f, (s, c) in r["family"].items()))
        print(f"  first quarter {r['first_quarter']:.1f}%  ->  last quarter {r['last_quarter']:.1f}%")
        print("  per cycle:    " + " ".join(f"{c:.0f}" for c in r["per_cycle"]))
    if len(rows) > 1:
        srs = [r["sr"] for r in rows]
        mean = sum(srs) / len(srs)
        sd = (sum((x - mean) ** 2 for x in srs) / (len(srs) - 1)) ** 0.5
        print(f"\nruns={len(rows)}  mean {mean:.2f}  sample SD {sd:.2f}  min {min(srs):.2f}  max {max(srs):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
