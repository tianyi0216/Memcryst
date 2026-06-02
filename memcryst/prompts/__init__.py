"""Prompt template loader.

We store prompts as plain markdown files in this package directory so we can
iterate on prompt wording without restarting Python. Use load_prompt with the
file stem to fetch the template, then call format on the returned string.
"""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt template not found: {path}")
    return path.read_text()
