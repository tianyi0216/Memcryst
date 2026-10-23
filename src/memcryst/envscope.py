"""Environment-scoped memory: entries are stamped with the environment they were
written in and retrieval can be restricted to the current environment
(``MEMCRYST_ENV_SCOPED=1``). Stamping is always on; filtering is opt-in.
"""
import os


_SUBSCOPE = ""   # e.g. the current conversation/stream id inside a QA run


def set_subscope(s: str) -> None:
    """Set by the env at reset. Content stores (facts, raw entries) are stamped
    down to this granularity when MEMCRYST_STREAM_SCOPED=1; skills stay
    env-level (answering strategies transfer across streams, facts do not)."""
    global _SUBSCOPE
    _SUBSCOPE = (s or "").strip().lower()


def stream_scoped() -> bool:
    return os.environ.get("MEMCRYST_STREAM_SCOPED", "0").strip().lower() \
        not in ("", "0", "false", "no")


def current_env(content: bool = False) -> str:
    base = os.environ.get("MEMCRYST_ENV", "alfworld").strip().lower()
    if content and stream_scoped() and _SUBSCOPE:
        return f"{base}:{_SUBSCOPE}"
    return base


def scoped() -> bool:
    return os.environ.get("MEMCRYST_ENV_SCOPED", "0").strip().lower() \
        not in ("", "0", "false", "no")


def effective_env(stamp: str) -> str:
    return (stamp or os.environ.get("MEMCRYST_LEGACY_ENV", "alfworld")).strip().lower()


def env_ok(stamp: str) -> bool:
    """True if an item with this env stamp may be retrieved right now.
    Sub-scoped stamps ("taskstream:3") require the matching subscope; base
    stamps ("taskstream") pass anywhere inside their env. With
    MEMCRYST_STREAM_SCOPED unset this reduces exactly to the old check."""
    if not scoped():
        return True
    e = effective_env(stamp)
    return e in (current_env(content=True), current_env())
