# Data preparation

## ALFWorld (primary benchmark)

```bash
bash data/prepare_alfworld.sh            # pip install alfworld + alfworld-download into data/raw/alfworld
export ALFWORLD_DATA=$PWD/data/raw/alfworld
```

The lifelong stream is the `train` split played in the environment's fixed
order (the first 600 games for the 20 x 30 configuration). Held-out probes use
`valid_unseen` (`eval_out_of_distribution`).

### Task-order manifests

`data/manifests/alf_order/` holds the canonical first-600 list and four
shuffled permutations of the same games (seeds 101, 202, 303, 404). They are
regenerated with

```bash
PYTHONPATH=src python data/build_order_manifests.py --n 600 --seeds 101 202 303 404
```

and selected through `env.order_manifest` in a config (see
`run_order_robustness.sh`). The environment verifies that a manifest is an
exact permutation of the canonical games and records the game file actually
played for every task.

## Conversational QA stream (DialSim)

`data/processed/dialsim/dialsim_shuf.json` is the event stream used for the
conversational-memory experiments: 200 dated sessions interleaved with 957
multiple-choice questions, derived from the DialSim benchmark (Friends, with
consistently renamed characters). It is a JSON list of events:

```json
{"kind": "session",  "stream": "0", "date": "September 22, 1994", "text": "Elena: ..."}
{"kind": "question", "stream": "0", "category": "ans_w_time_q", "question": "...\nOptions: A / B / ...", "answer": "..."}
```

Questions are scored by normalized token F1 against `answer` (success = F1 >= 0.5).
Any stream in this format can be run through the same pipeline by pointing
`env.stream_file` at it.
