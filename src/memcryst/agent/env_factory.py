"""Environment factory: MEMCRYST_ENV selects the environment (alfworld | taskstream)."""
import os

from memcryst.agent.alfworld_env import AlfworldEnv


def make_env(split: str, max_steps: int):
    kind = os.environ.get("MEMCRYST_ENV", "alfworld").strip().lower()
    if kind == "taskstream":
        from memcryst.agent.taskstream_env import TaskstreamEnv
        return TaskstreamEnv(split=split, max_steps=max_steps)
    if kind != "alfworld":
        raise ValueError(f"unsupported MEMCRYST_ENV={kind!r}; this release ships alfworld and taskstream")
    return AlfworldEnv(split=split, max_steps=max_steps)
