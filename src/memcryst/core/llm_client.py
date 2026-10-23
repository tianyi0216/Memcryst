"""Synchronous OpenAI compatible chat client.

A synchronous client (the sleep loop issues one call at a time and reads the
result before proceeding) with retries, temperature normalization and
explicit max_tokens handling.

Set OPENAI_API_KEY before constructing the client. Use a custom base_url to
hit Azure, Anthropic via a proxy, or a local server such as vLLM.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def _normalize_temperature(model: str, requested: float) -> float:
    name = (model or "").lower().strip()
    # Reasoning models reject non default temperature.
    if name.startswith("o1") or name.startswith("o3") or name.startswith("o4"):
        return 1.0
    return requested


class LLMClient:
    """Thin wrapper around the synchronous OpenAI SDK with retries.

    The default model is gpt-4o-mini because it is cheap, fast, and a good
    fit for the synthesizer and audit prompts. Pass a different model for the
    wake time agent.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 4096,
        temperature: float = 0.3,
        max_retries: int = 5,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required, install with pip install openai"
            ) from exc

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        env_base = os.environ.get("OPENAI_BASE_URL")
        # API-model routing: gpt-*/o*-family models must hit the real OpenAI
        # endpoint even when OPENAI_BASE_URL points at a local vLLM server
        # (teacher-role setups where agent=local, writeback/synth=API).
        if base_url is None and env_base and any(
            model.lower().startswith(pfx) for pfx in ("gpt-", "o1", "o3", "o4", "chatgpt")
        ):
            # Must be EXPLICIT: with base_url=None the OpenAI SDK falls back to
            # the same OPENAI_BASE_URL env var we are trying to bypass.
            env_base = "https://api.openai.com/v1"
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url or env_base,
        )

    def chat(
        self,
        messages: list[dict[str, str]] | list[ChatMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request and return the response text."""
        msgs = [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages]
        chosen_model = model or self.model
        chosen_temp = _normalize_temperature(chosen_model, temperature if temperature is not None else self.temperature)
        chosen_max = max_tokens if max_tokens is not None else self.max_tokens

        # Token cap parameter naming differs across backends. OpenAI reasoning
        # models such as o1/o3/o4 require max_completion_tokens. vLLM, Ollama,
        # most Anthropic compatibility proxies, and older OpenAI chat endpoints
        # still expect max_tokens. We pick the right one based on the model name.
        name_lc = (chosen_model or "").lower()
        reasoning_model = name_lc.startswith(("o1", "o3", "o4"))
        token_key = "max_completion_tokens" if reasoning_model else "max_tokens"

        params: dict[str, Any] = {
            "model": chosen_model,
            "messages": msgs,
            "temperature": chosen_temp,
            token_key: chosen_max,
        }
        params.update(kwargs)

        # Qwen3 thinking mode is disabled through vLLM's chat_template_kwargs so
        # short action generations are not truncated mid-thought.
        if "qwen3" in name_lc:
            extra_body = dict(params.get("extra_body") or {})
            ctk = dict(extra_body.get("chat_template_kwargs") or {})
            ctk.setdefault("enable_thinking", False)
            extra_body["chat_template_kwargs"] = ctk
            params["extra_body"] = extra_body

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**params)
                if resp.choices and resp.choices[0].message:
                    return resp.choices[0].message.content or ""
                return ""
            except Exception as exc:
                last_err = exc
                backoff = min(2 ** attempt, 30)
                logger.warning(
                    "LLM call failed on attempt %d, retrying in %ds, err=%r",
                    attempt + 1,
                    backoff,
                    exc,
                )
                time.sleep(backoff)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts") from last_err
