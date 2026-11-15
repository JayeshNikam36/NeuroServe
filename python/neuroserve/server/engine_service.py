"""Owns the engine and tokenizer; turns chat messages or prompts into a completion."""

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
class GenerationResult:
    text: str              # decoded returned tokens, without the stop token
    token_ids: list[int]   # returned tokens, including the stop token when one was hit
    prompt_tokens: int
    generated_tokens: int  # tokens the engine produced (>= len(token_ids) until Phase 2)
    finish_reason: str     # "stop" or "length"
    e2e_ms: float

    @property
    def completion_tokens(self) -> int:
        return len(self.token_ids)


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

    def chat(self, messages: list[dict[str, Any]], max_new_tokens: int, *,
             ignore_eos: bool = False) -> GenerationResult:
        prompt_ids = self._tokenizer.encode_chat(messages, date_string=self._config.model.chat_date_string)
        return self._run(prompt_ids, max_new_tokens, ignore_eos)

    def complete(self, prompt: str | list[int], max_new_tokens: int, *,
                 ignore_eos: bool = False) -> GenerationResult:
        if isinstance(prompt, str):
            prompt_ids = self._tokenizer.encode(prompt)  # adds <|begin_of_text|>
        else:
            vocab = self._tokenizer.vocab_size
            bad = next((t for t in prompt if not 0 <= t < vocab), None)
            if bad is not None:
                raise ValueError(f"token id {bad} is outside the vocabulary (0..{vocab - 1})")
            prompt_ids = list(prompt)  # token ids are used verbatim
        return self._run(prompt_ids, max_new_tokens, ignore_eos)

    def _run(self, prompt_ids: list[int], max_new_tokens: int, ignore_eos: bool) -> GenerationResult:
        limit = self._config.engine.max_prompt_tokens
        if len(prompt_ids) > limit:
            raise ContextLengthExceeded(f"the prompt has {len(prompt_ids)} tokens; the limit is {limit}")

        start = time.perf_counter()
        with self._lock:
            out = list(self._engine.generate(prompt_ids, max_new_tokens=max_new_tokens))
        e2e_ms = (time.perf_counter() - start) * 1000.0

        stop_at = None
        if not ignore_eos:
            stop_at = next((i for i, t in enumerate(out) if t in self._tokenizer.stop_token_ids), None)
        if stop_at is None:
            returned, finish_reason, text_ids = out, "length", out
        else:
            returned, finish_reason, text_ids = out[:stop_at + 1], "stop", out[:stop_at]

        return GenerationResult(
            text=self._tokenizer.decode(text_ids),
            token_ids=returned,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(out),
            finish_reason=finish_reason,
            e2e_ms=e2e_ms,
        )
