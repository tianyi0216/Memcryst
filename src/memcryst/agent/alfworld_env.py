"""Thin adapter around ALFWorld's text environment (AlfredTWEnv).

ALFWorld is a text-based household environment with six task families
(pick-and-place, pick-two, clean, heat, cool, examine). Each task is a
TextWorld game whose initial observation names the task and the visible
receptacles. We use the text-only flavour.

Install with ``pip install alfworld && alfworld-download`` and set
``ALFWORLD_DATA`` to the download directory before importing this module.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# ALFWorld task family tags as they appear in game file names.
TASK_FAMILIES = {
    "clean": "pick_clean_then_place_in_recep",
    "heat": "pick_heat_then_place_in_recep",
    "cool": "pick_cool_then_place_in_recep",
    "examine": "look_at_obj_in_light",
    "place": "pick_and_place_simple",
    "place_two": "pick_two_obj_and_place",
}


# Map ALFWorld family code back to our capability verb_family enum.
FAMILY_TO_VERB: dict[str, str] = {
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "look_at_obj_in_light": "examine",
    "pick_and_place_simple": "move",
    "pick_two_obj_and_place": "move",
}


@dataclass
class TaskMeta:
    task_idx: int
    family: str          # ALFWorld family code, e.g. "pick_clean_then_place_in_recep"
    verb_family: str     # canonical verb family, e.g. "clean"
    game_file: str = ""  # for debugging, the .tw-pddl file


@dataclass
class ResetResult:
    obs: str
    task_description: str    # parsed from the initial obs "Your task is to: ..."
    valid_actions: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    obs: str
    reward: float
    done: bool
    success: bool            # task solved per ALFWorld signal
    valid_actions: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


def _default_config_path() -> Path:
    """Where to look for an ALFWorld YAML config. We ship a minimal one
    next to this file because the package config defaults to embodied mode."""
    return Path(__file__).parent / "alfworld_config.yaml"


def _ensure_default_config() -> Path:
    path = _default_config_path()
    if path.exists():
        return path
    # Text-only config compatible with alfworld 0.4.x. The schema requires
    # both `dagger` and `rl` blocks even when we just want to eval, because
    # AlfredTWEnv.init_env keys into one of them based on training_method.
    cfg = {
        "dataset": {
            "data_path": "$ALFWORLD_DATA/json_2.1.1/train",
            "eval_id_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_seen",
            "eval_ood_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_unseen",
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {
            "domain": "$ALFWORLD_DATA/logic/alfred.pddl",
            "grammar": "$ALFWORLD_DATA/logic/alfred.twl2",
        },
        "env": {
            "type": "AlfredTWEnv",
            "regen_game_files": False,
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
            "hybrid": {
                "start_eps": 100000,
                "thor_prob": 0.5,
                "eval_mode": "tw",
            },
            "thor": {
                "screen_width": 300,
                "screen_height": 300,
                "smooth_nav": False,
                "save_frames_to_disk": False,
                "save_frames_path": "./videos/",
            },
        },
        "controller": {
            "type": "oracle",
            "debug": False,
            "load_receps": True,
        },
        "general": {
            "training_method": "dagger",
            "save_path": "./training",
            "observation_pool_capacity": 3,
            "hide_init_receptacles": False,
            "task_config_file": "",
            "training": {
                "batch_size": 1,
                "max_episode": 50000,
                "smoothing_eps": 0.1,
                "optimizer": {
                    "learning_rate": 0.001,
                    "clip_grad_norm": 5,
                },
            },
            "checkpoint": {
                "report_frequency": 1000,
                "experiment_tag": "test",
                "load_pretrained": False,
                "load_from_tag": "not loading anything",
            },
            "model": {
                "encoder_layers": 1,
                "decoder_layers": 1,
                "encoder_conv_num": 5,
                "block_hidden_dim": 64,
                "n_heads": 1,
                "dropout": 0.1,
                "block_dropout": 0.1,
                "recurrent": True,
            },
        },
        "rl": {
            "action_space": "admissible",
            "max_target_length": 20,
            "beam_width": 10,
            "generate_top_k": 3,
            "training": {
                "max_nb_steps_per_episode": 50,
                "learn_start_from_this_episode": 0,
                "target_net_update_frequency": 500,
            },
            "replay": {
                "accumulate_reward_from_final": True,
                "count_reward_lambda": 0.0,
                "novel_object_reward_lambda": 0.0,
                "discount_gamma_game_reward": 0.9,
                "discount_gamma_count_reward": 0.5,
                "discount_gamma_novel_object_reward": 0.5,
                "replay_memory_capacity": 500000,
                "replay_memory_priority_fraction": 0.5,
                "update_per_k_game_steps": 5,
                "replay_batch_size": 64,
                "multi_step": 3,
                "replay_sample_history_length": 4,
                "replay_sample_update_from": 2,
            },
            "epsilon_greedy": {
                "noisy_net": False,
                "epsilon_anneal_episodes": 1000,
                "epsilon_anneal_from": 0.3,
                "epsilon_anneal_to": 0.1,
            },
        },
        "dagger": {
            "action_space": "generation",
            "max_target_length": 20,
            "beam_width": 10,
            "generate_top_k": 5,
            "unstick_by_beam_search": False,
            "training": {
                "max_nb_steps_per_episode": 50,
            },
            "fraction_assist": {
                "fraction_assist_anneal_episodes": 50000,
                "fraction_assist_anneal_from": 1.0,
                "fraction_assist_anneal_to": 0.0,
            },
            "fraction_random": {
                "fraction_random_anneal_episodes": 0,
                "fraction_random_anneal_from": 0.0,
                "fraction_random_anneal_to": 0.0,
            },
            "replay": {
                "replay_memory_capacity": 500000,
                "update_per_k_game_steps": 5,
                "replay_batch_size": 64,
                "replay_sample_history_length": 4,
                "replay_sample_update_from": 2,
            },
        },
        "vision_dagger": {
            "model_type": "resnet",
            "resnet_fc_dim": 64,
            "maskrcnn_top": 10,
            "use_exploration_frame_feats": False,
            "sequence_aggregation_method": "average",
        },
    }
    path.write_text(yaml.safe_dump(cfg))
    return path


_TASK_RE = re.compile(r"your task is to:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def _extract_task_description(obs: str) -> str:
    match = _TASK_RE.search(obs or "")
    if not match:
        return ""
    return match.group(1).strip().rstrip(".")


def _detect_family(game_file: str) -> tuple[str, str]:
    """Detect ALFWorld family code and canonical verb_family from the game file path.

    Warning: game file names are unreliable for verb_family classification in
    ALFWorld 0.4.x because the file describes the GAME, not the task. Two
    trials of the same game can have different tasks. Use
    family_from_task_description() instead for actual verb classification.
    """
    name = (game_file or "").lower()
    for fam_code, verb in FAMILY_TO_VERB.items():
        if fam_code in name:
            return fam_code, verb
    return "unknown", "unknown"


# Alternate envs (e.g. WebShop) register task_description -> family here so
# every existing family_from_task_description() call site works unchanged.
_FAMILY_OVERRIDE: dict = {}


def register_family_override(task_description: str, family: str) -> None:
    _FAMILY_OVERRIDE[(task_description or "").lower().strip()] = family


def family_from_task_description(task: str) -> str:
    """Classify an ALFWorld task description into a verb_family.

    Reliable: parses the actual goal text. Use this in eval, not the file path.
    Returns one of clean / heat / cool / examine / move / unknown.
    """
    text = (task or "").lower().strip()
    if not text:
        return "unknown"
    if _FAMILY_OVERRIDE:
        fam = _FAMILY_OVERRIDE.get(text)
        if fam:
            return fam
    # examine wins because look_at_obj tasks always contain "examine" or "look at"
    if "examine" in text or "look at" in text:
        return "examine"
    # clean modifier or verb
    if re.search(r"\bclean\b", text):
        return "clean"
    # heat: "hot X" or "heat X" or "warm X" — but NOT just "microwave" target
    if re.search(r"\bhot\b", text) or re.search(r"\bheat\b", text) or re.search(r"\bwarm\b", text):
        return "heat"
    # cool: "cool X" or "cold X" or "chill X" — modifier, not destination
    if re.search(r"\bcool\b", text) or re.search(r"\bcold\b", text) or re.search(r"\bchill\b", text):
        return "cool"
    # default: plain pick-and-place
    return "move"


class AlfworldEnv:
    """Wrapper around the ALFWorld TextWorld backend.

    Usage:
        env = AlfworldEnv(split="eval_out_of_distribution")
        reset = env.reset(task_idx=0)
        print(reset.task_description)
        step = env.step("look")
    """

    def __init__(
        self,
        split: str = "eval_out_of_distribution",
        config_path: Optional[Path] = None,
        max_steps: int = 50,
    ) -> None:
        try:
            from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "alfworld is not installed. Run:\n"
                "  pip install alfworld\n"
                "  alfworld-download\n"
                "  export ALFWORLD_DATA=\"$HOME/.cache/alfworld\"\n"
                f"Underlying error: {exc!r}"
            ) from exc

        if not os.environ.get("ALFWORLD_DATA"):
            logger.warning(
                "ALFWORLD_DATA env var not set, ALFWorld will likely fail to locate data."
            )

        self.split = split
        self.max_steps = max_steps
        self.config_path = Path(config_path) if config_path else _ensure_default_config()
        self._cfg = self._load_config(self.config_path)
        self._raw_env = None
        self._wrapped_env = None
        self._game_files: list[str] = []
        self._steps_in_episode = 0
        self._order_manifest: Optional[list[str]] = None   # explicit play order (see _init_env)
        self._order_seed = None
        self.last_gamefile: Optional[str] = None            # gamefile actually loaded by the last reset()
        self._init_env()

    @staticmethod
    def _load_config(path: Path) -> dict:
        text = path.read_text()
        text = os.path.expandvars(text)
        return yaml.safe_load(text)

    def _init_env(self) -> None:
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

        self._raw_env = AlfredTWEnv(self._cfg, train_eval=self.split)
        # DETERMINISTIC game order across runs/processes. ALFWorld shuffles the
        # game-file list per process (no fixed seed), so task_idx i would map to a
        # DIFFERENT game in different runs -> cross-run / cross-method comparisons
        # become apples-to-oranges. Sort the list BEFORE init_env so every run sees
        # the identical ordered game set and task_idx i is the same game everywhere.
        try:
            self._raw_env.game_files = sorted(self._raw_env.game_files)
        except Exception:
            logger.warning("could not sort game_files; eval game order may be non-deterministic")
        mpath = os.environ.get("MEMCRYST_ALF_ORDER_MANIFEST", "").strip()
        if mpath and self.split == "train":
            import json as _json
            import numpy as _np
            man = _json.loads(Path(mpath).read_text())
            root = os.environ.get("ALFWORLD_DATA", "")
            files = [g if os.path.isabs(g) else os.path.join(root, g) for g in man["gamefiles"]]
            canon = list(self._raw_env.game_files)            # sorted above
            _np.random.RandomState(1234).shuffle(canon)       # == TextworldBatchGymEnv.seed(1234)
            canon = canon[:len(files)]
            if len(set(files)) != len(files):
                raise RuntimeError("order manifest contains duplicate games")
            if sorted(canon) != sorted(files):
                missing = sorted(set(canon) - set(files))[:3]
                extra = sorted(set(files) - set(canon))[:3]
                raise RuntimeError(f"order manifest is not a permutation of the canonical first {len(files)} games "
                                   f"(missing e.g. {missing}; extra e.g. {extra})")
            self._order_manifest = files
            self._order_seed = man.get("order_seed")
        # init_env returns a wrapped env. batch_size=1 because we run one task at a time.
        self._wrapped_env = self._raw_env.init_env(batch_size=1)
        if self._order_manifest is not None:
            inner = self._wrapped_env
            seen = 0
            while inner is not None and not hasattr(inner, "_gamefiles_iterator") and seen < 8:
                inner = getattr(inner, "env", None) or getattr(inner, "unwrapped", None)
                seen += 1
            if inner is None or not hasattr(inner, "_gamefiles_iterator"):
                raise RuntimeError("cannot reach TextworldBatchGymEnv._gamefiles_iterator to impose the manifest order")
            inner._gamefiles_iterator = iter(list(self._order_manifest))   # strict: StopIteration on the (N+1)-th reset
            logger.info("ALFWorld order manifest active: %d games, order_seed=%s", len(self._order_manifest), self._order_seed)
        # Game files give us per-task family metadata.
        self._game_files = list(getattr(self._raw_env, "game_files", []) or [])
        logger.info(
            "AlfworldEnv ready, split=%s, num_games=%d",
            self.split,
            len(self._game_files),
        )

    # ---- task enumeration ----

    def list_tasks(self, limit: Optional[int] = None) -> list[TaskMeta]:
        out: list[TaskMeta] = []
        for idx, gf in enumerate(self._game_files):
            fam_code, verb = _detect_family(gf)
            out.append(
                TaskMeta(
                    task_idx=idx,
                    family=fam_code,
                    verb_family=verb,
                    game_file=gf,
                )
            )
            if limit is not None and len(out) >= limit:
                break
        return out

    # ---- gym-style API ----

    def reset(self, task_idx: Optional[int] = None) -> ResetResult:
        """Reset to a specific task. ALFWorld iterates internally so we just
        skip forward until we reach the requested task."""
        if task_idx is not None:
            # ALFWorld 0.3.x: setting `game_idx` before reset jumps to that game.
            try:
                self._raw_env.game_idx = task_idx
            except Exception:
                logger.debug("could not set game_idx, falling back to sequential reset")
        # ALFWorld's expert can raise inside reset on some games; skip to the next game.
        for _retry in range(4):
            try:
                obs_list, info = self._wrapped_env.reset()
                break
            except Exception as e:
                logger.warning("env.reset raised %s on game %s; skipping to next game",
                               type(e).__name__, getattr(self._raw_env, "game_idx", "?"))
                try:
                    self._raw_env.game_idx = int(getattr(self._raw_env, "game_idx", 0)) + 1
                except Exception:
                    pass
        else:
            raise RuntimeError("env.reset failed on 4 consecutive games")
        obs = obs_list[0] if isinstance(obs_list, (list, tuple)) else obs_list
        task_desc = _extract_task_description(obs)
        valid = self._extract_valid_actions(info)
        self._steps_in_episode = 0
        try:   # AlfredTWEnv requests extras=["gamefile"]: record what was ACTUALLY loaded
            _gf = info.get("extra.gamefile") if isinstance(info, dict) else None
            self.last_gamefile = (_gf[0] if isinstance(_gf, (list, tuple)) and _gf else _gf) or None
        except Exception:
            self.last_gamefile = None
        return ResetResult(
            obs=obs,
            task_description=task_desc,
            valid_actions=valid,
            info=self._normalize_info(info),
        )

    def step(self, action: str) -> StepResult:
        action_batch = [action]
        try:
            obs_list, reward_list, done_list, info = self._wrapped_env.step(action_batch)
        except Exception as e:
            # ALFWorld's expert can raise on some game states; end the episode as failed.
            logger.warning("env.step raised %s (%s); ending episode as failed", type(e).__name__, e)
            self._steps_in_episode += 1
            return StepResult(
                obs="", reward=0.0, done=True, success=False,
                valid_actions=[], info={"won": False, "env_error": str(e)},
            )
        obs = obs_list[0] if isinstance(obs_list, (list, tuple)) else obs_list
        reward = float(reward_list[0]) if isinstance(reward_list, (list, tuple)) else float(reward_list)
        done = bool(done_list[0]) if isinstance(done_list, (list, tuple)) else bool(done_list)
        success = bool(info.get("won", [False])[0]) if isinstance(info.get("won"), list) else bool(info.get("won", False))
        valid = self._extract_valid_actions(info)
        self._steps_in_episode += 1
        if self._steps_in_episode >= self.max_steps and not done:
            done = True
        return StepResult(
            obs=obs,
            reward=reward,
            done=done,
            success=success,
            valid_actions=valid,
            info=self._normalize_info(info),
        )

    # ---- helpers ----

    @staticmethod
    def _extract_valid_actions(info: dict) -> list[str]:
        if not isinstance(info, dict):
            return []
        actions = info.get("admissible_commands") or info.get("valid_actions") or []
        if actions and isinstance(actions[0], list):
            actions = actions[0]
        return [str(a) for a in actions]

    @staticmethod
    def _normalize_info(info: Any) -> dict:
        if isinstance(info, dict):
            return info
        return {"raw": info}
