"""Config-driven launcher for the lifelong wake-sleep pipeline.

Reads a YAML config, resolves it into environment switches plus command-line
flags for ``memcryst.pipeline.lifelong``, records the resolved configuration
in the run directory, and streams the run log.

    python src/run.py --config configs/alfworld_full.yaml
    python src/run.py --config configs/alfworld_full.yaml --set run.name=my_run --set skills.top_k=1
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# config key -> environment switch read by the pipeline
ENV_MAP = {
    ("actor", "nav_hint"): "MEMCRYST_NAV_HINT",
    ("actor", "wm_enforce"): "MEMCRYST_WM_ENFORCE",
    ("actor", "qa_anscall"): "MEMCRYST_QA_ANSCALL",
    ("memory", "raw_layer"): "MEMCRYST_RAW_LAYER",
    ("memory", "raw_topk"): "MEMCRYST_RAW_TOPK",
    ("memory", "raw_recency"): "MEMCRYST_RAW_RECENCY",
    ("memory", "raw_deinstantiate"): "MEMCRYST_RAW_DEINST",
    ("memory", "fact_style"): "MEMCRYST_FACT_STYLE",
    ("memory", "writeback_verbatim"): "MEMCRYST_WB_VERBATIM",
    ("memory", "writeback_session_only"): "MEMCRYST_WB_SESSION_ONLY",
    ("memory", "hybrid_lexical"): "MEMCRYST_HYBRID_LEX",
    ("memory", "env_scoped"): "MEMCRYST_ENV_SCOPED",
    ("memory", "embed_model"): "MEMCRYST_EMBED_MODEL",
    ("memory", "semantic_profile"): "MEMCRYST_SEMANTIC_PROFILE",
    ("skills", "synth_prompt"): "MEMCRYST_SYNTH_PROMPT",
    ("skills", "gap_val_n"): "MEMCRYST_GAP_VAL_N",
    ("skills", "gap_val_start"): "MEMCRYST_GAP_VAL_START",
    ("skills", "freeze_after"): "MEMCRYST_FREEZE_SKILLS_AFTER",
    ("skills", "dedup"): "MEMCRYST_SKILL_DEDUP",
    ("skills", "fallback"): "MEMCRYST_SKILL_FALLBACK",
    ("skills", "failure_memory"): "MEMCRYST_SKILL_FAILMEM",
    ("feedback", "memory_return"): "MEMCRYST_FB_MEM",
    ("feedback", "skill_return"): "MEMCRYST_FB_SKILL",
}


def deep_set(cfg: dict, dotted: str, value: str) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(value)


def resolve(cfg: dict) -> tuple[dict, list[str], Path]:
    run = cfg.get("run", {})
    model = cfg.get("model", {})
    env = cfg.get("env", {})
    stream = cfg.get("stream", {})
    memory = cfg.get("memory", {})
    skills = cfg.get("skills", {})
    actor = cfg.get("actor", {})

    out_dir = (ROOT / run.get("out_root", "runs") / run["name"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("memory", "skills", "semantic"):          # empty initial bank
        (out_dir / "empty_input" / sub).mkdir(parents=True, exist_ok=True)

    e = {k: v for k, v in os.environ.items() if not k.startswith("MEMCRYST_")}
    e["PYTHONPATH"] = str(SRC) + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    e["PYTHONUNBUFFERED"] = "1"
    if model.get("base_url"):
        e["OPENAI_BASE_URL"] = model["base_url"]
    key_env = model.get("api_key_env", "OPENAI_API_KEY")
    e["OPENAI_API_KEY"] = os.environ.get(key_env) or e.get("OPENAI_API_KEY") or "EMPTY"

    kind = env.get("kind", "alfworld")
    e["MEMCRYST_ENV"] = kind
    if kind == "alfworld":
        data = env.get("alfworld_data") or os.environ.get("ALFWORLD_DATA")
        if not data:
            sys.exit("set env.alfworld_data in the config or export ALFWORLD_DATA")
        e["ALFWORLD_DATA"] = str(Path(data).expanduser().resolve())
        if env.get("order_manifest"):
            e["MEMCRYST_ALF_ORDER_MANIFEST"] = str((ROOT / env["order_manifest"]).resolve())
    elif kind == "taskstream":
        e["MEMCRYST_TASKSTREAM_DATA"] = str((ROOT / env["stream_file"]).resolve())
        e["MEMCRYST_TASKSTREAM_KIND"] = str(env.get("stream_kind", "qa"))
        e["MEMCRYST_TASKSTREAM_EVAL"] = str(env.get("stream_eval", "f1"))
    else:
        sys.exit(f"unsupported env.kind: {kind}")

    mode = actor.get("mode", "capability_plus_semantic")
    e["MEMCRYST_WAKE_MODE"] = mode
    e["MEMCRYST_EVAL_MODE"] = mode
    for (sec, key), var in ENV_MAP.items():
        val = cfg.get(sec, {}).get(key)
        if val is None:
            continue
        e[var] = str(int(val)) if isinstance(val, bool) else str(val)
    for k, v in (cfg.get("extra_env") or {}).items():
        e[str(k)] = str(v)

    name = model["name"]
    roles = model.get("roles") or {}
    argv = [
        sys.executable, "-u", "-m", "memcryst.pipeline.lifelong",
        "--input-dir", str(out_dir / "empty_input"),
        "--output-dir", str(out_dir / "bank"),
        "--output-json", str(out_dir / "lifelong.json"),
        "--num-cycles", str(stream.get("num_cycles", 20)),
        "--tasks-per-cycle", str(stream.get("tasks_per_cycle", 30)),
        "--split", str(env.get("split", "train")),
        "--eval-split", str(env.get("eval_split", env.get("split", "train"))),
        "--eval-tasks", str(env.get("eval_tasks", 1)),
        "--eval-task-start", str(env.get("eval_task_start", 0)),
        "--eval-every", str(env.get("eval_every", 999)),
        "--max-steps", str(env.get("max_steps", 30)),
        "--m3-mode", str(skills.get("m3_mode", "off")),
        "--refine" if skills.get("refine", True) else "--no-refine",
        "--top-k-memory", str(memory.get("top_k", 4)),
        "--top-k-skills", str(skills.get("top_k", 2)),
        "--min-cluster-size", str(skills.get("min_cluster_size", 3)),
        "--agent-model", roles.get("agent") or name,
        "--writeback-model", roles.get("writeback") or name,
        "--synth-model", roles.get("synth") or name,
        "--audit-model", roles.get("audit") or name,
    ]
    return e, argv, out_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set skills.top_k=1")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved command and exit")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for item in args.set:
        k, _, v = item.partition("=")
        deep_set(cfg, k, v)
    env, argv, out_dir = resolve(copy.deepcopy(cfg))

    resolved = {"config": cfg, "command": argv,
                "env": {k: v for k, v in env.items() if k.startswith(("MEMCRYST_", "ALFWORLD_", "OPENAI_BASE_URL"))}}
    (out_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=1))
    print("run dir:", out_dir)
    print("command:", " ".join(argv))
    if args.dry_run:
        return 0

    log = open(out_dir / "run.log", "a")
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log.write(line)
    rc = proc.wait()
    log.close()
    if rc == 0 and (out_dir / "lifelong.json").exists():
        subprocess.run([sys.executable, str(SRC / "metrics.py"), str(out_dir / "lifelong.json")], env=env)
    return rc


if __name__ == "__main__":
    sys.exit(main())
