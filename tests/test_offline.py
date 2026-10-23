"""Offline checks that need no model or benchmark data.

    PYTHONPATH=src pytest -q tests
"""
import os
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from memcryst.utils.text_metrics import token_f1  # noqa: E402


def test_token_f1():
    assert token_f1("Dad's credit card", "Dad's credit card") == 1.0
    assert token_f1("the credit card", "credit card") == 1.0
    assert token_f1("Paris", "London") == 0.0


def test_config_resolution_memory_only(monkeypatch):
    from src import run as runner  # type: ignore
    cfg = yaml.safe_load(open(ROOT / "configs" / "alfworld_memory_only.yaml"))
    with tempfile.TemporaryDirectory() as d:
        cfg["run"]["out_root"] = d
        monkeypatch.setenv("ALFWORLD_DATA", d)
        env, argv, out_dir = runner.resolve(cfg)
    assert env["MEMCRYST_FREEZE_SKILLS_AFTER"] == "-1"
    assert env["MEMCRYST_FB_SKILL"] == "0" and env["MEMCRYST_FB_MEM"] == "1"
    assert "--no-refine" in argv and argv[argv.index("--top-k-skills") + 1] == "0"
    assert argv[argv.index("--top-k-memory") + 1] == "4"


def test_skill_store_roundtrip():
    from memcryst.skill_bank import Skill, SkillStore
    with tempfile.TemporaryDirectory() as d:
        store = SkillStore(store_dir=Path(d))
        sk = Skill(name="clean-and-place", description="Clean an object and place it.",
                   body="## When to use\n- clean tasks\n\n## Procedure\n- go to sink\n- clean object\n")
        store.add(sk)
        assert len(SkillStore(store_dir=Path(d))) == 1
        reloaded = next(iter(SkillStore(store_dir=Path(d))))
        assert reloaded.name == "clean-and-place" and "Procedure" in reloaded.body


def test_manifests_are_permutations():
    import json
    base = ROOT / "data" / "manifests" / "alf_order"
    canon = json.load(open(base / "canonical_first600.json"))["gamefiles"]
    assert len(canon) == 600 == len(set(canon))
    for seed in (101, 202, 303, 404):
        m = json.load(open(base / f"order_seed{seed}.json"))
        assert m["order_seed"] == seed and sorted(m["gamefiles"]) == sorted(canon)
