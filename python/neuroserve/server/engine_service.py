"""Owns the engine and tokenizer; turns chat messages into a completion."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

from ..tokenizer import ChatTokenizer
from .config import ServerConfig


class GenerationEngine(Protocol):
    """What the service needs from an engine (the real neuroserve.Engine, or a test fake)."""

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> list[int]: ...

    def kv_stats(self) -> dict[str, Any]: ...


class ContextLengthExceeded(ValueError):
    """The prompt is longer than the engine accepts."""


@dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int  # tokens returned to the client, including the stop token
    generated_tokens: int   # tokens the engine produced (>= completion_tokens until Phase 2)
    finish_reason: str      # "stop" or "length"
    e2e_ms: float


class EngineService:
    """One generation at a time: the engine is not thread-safe.

    Continuous batching (Phase 2) replaces the lock with a scheduler.
    """

    def __init__(self, config: ServerConfig, engine: GenerationEngine, tokenizer: ChatTokenizer,
                 device_name: str) -> None:
        self._config = config
        self._engine = engine
        self._tokenizer = tokenizer
        self._device_name = device_name
        self._lock = threading.Lock()

    @classmethod
    def load(cls, config: ServerConfig) -> "EngineService":
        """Loads the real model onto the GPU (takes a few seconds)."""
        from .. import Engine, device_info

        e = config.engine
        engine = Engine(
            str(config.model.weights_dir),
            max_prompt_tokens=e.max_prompt_tokens,
            num_pages=e.num_pages,
            max_batch=e.max_batch,
            max_pages_per_sequence=e.max_pages_per_sequence,
            device_index=e.device_index,
        )
        tokenizer = ChatTokenizer(config.model.tokenizer_dir)
        return cls(config, engine, tokenizer, device_info(e.device_index)["name"])

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def tokenizer(self) -> ChatTokenizer:
        return self._tokenizer

    def kv_stats(self) -> dict[str, Any]:
        return self._engine.kv_stats()

    def chat(self, messages: list[dict[str, Any]], max_new_tokens: int) -> ChatResult:
        prompt_ids = self._tokenizer.encode_chat(messages)
        limit = self._config.engine.max_prompt_tokens
        if len(prompt_ids) > limit:
            raise ContextLengthExceeded(f"the prompt has {len(prompt_ids)} tokens; the limit is {limit}")

        start = time.perf_counter()
        with self._lock:
            out = list(self._engine.generate(prompt_ids, max_new_tokens=max_new_tokens))
        e2e_ms = (time.perf_counter() - start) * 1000.0

        stop_at = next((i for i, t in enumerate(out) if t in self._tokenizer.stop_token_ids), None)
        if stop_at is None:
            kept, finish_reason, completion_tokens = out, "length", len(out)
        else:
            kept, finish_reason, completion_tokens = out[:stop_at], "stop", stop_at + 1

        return ChatResult(
            text=self._tokenizer.decode(kept),
            prompt_tokens=len(prompt_ids),
            completion_tokens=completion_tokens,
            generated_tokens=len(out),
            finish_reason=finish_reason,
            e2e_ms=e2e_ms,
        )
