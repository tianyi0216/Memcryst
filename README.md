# MEMCRYST v1 PoC

Memory crystallization for self improving language model agents. Training free instantiation.

## What this is

This is the proof of concept implementation for MEMCRYST v1. The system has an episodic memory bank and a procedural skill bank that interact through three sub mechanisms at sleep time. A pattern miner clusters memory entries by task signature. A synthesizer turns dense reward positive clusters into draft skill artifacts. A fresh agent audit gates draft skills before they enter the skill bank. A feedback loop traces skill failures back to source memory entries and flags them as unreliable.

The training free version uses only language model API calls. Default backbone is GPT-4.1-mini for the agent and GPT-4o-mini for the synthesizer and audit. No GPU required.

## Module map

```
memcryst/
  memory_bank/       fork of live-evo experience_manager with ancestry pointer
  skill_bank/        MUSE style SKILL.md storage with retrieval
  core/              shared utilities, llm_client
  sleep/             M1 miner, M2 synthesizer, M2 audit, M3 feedback
  agent/             wake time agent loop, alfworld prompts
  prompts/           markdown prompt templates loaded at runtime
  examples/          synthetic pilot, alfworld run
  tests/             unit tests
```

## Quick start

```bash
pip install -e .
export OPENAI_API_KEY="sk-..."
python -m memcryst.examples.synthetic_pilot
```

The synthetic pilot does the end to end sanity check without an environment. It seeds the memory bank with five hand crafted entries, runs the pattern miner, synthesizes a candidate skill, runs the audit, and prints the verdict.

## Borrowed code map

| Module | Origin | Why |
|---|---|---|
| memory_bank | live-evo-0.1.0 experience_manager | 6 field schema, weight update, JSONL persistence |
| skill_bank | MetaClaw skill_manager + MUSE SKILL.md examples | YAML frontmatter, category field |
| core/llm_client | SkillClaw evolve_server llm_client | Async OpenAI client with retry |
| sleep/m2_synthesizer | SkillClaw summarizer prompt pattern | Trajectory analysis prompt structure |
| sleep/m2_audit | SkillClaw skill_verifier | JSON verdict structure, grounding checks |
| agent/alfworld_prompts | Skill1 alfworld prompts | ALFWORLD_TEMPLATE with skill injection slot |

## Status

Phase 0 hello world target. End to end synthetic pilot should run in under one minute on a laptop with API access.
