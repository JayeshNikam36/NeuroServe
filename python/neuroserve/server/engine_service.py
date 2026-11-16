"""Owns the engine and tokenizer; turns chat messages or prompts into a completion."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from ..tokenizer import ChatTokenizer
from .config import ServerConfig
from .logging_setup import request_id_var


class EngineOutput(Protocol):
    """One finished request, as the engine reports it (neuroserve.GenerationResult or a test fake)."""

    request_id: str      # the engine's id, e.g. "ns-000042"
    tokens: list[int]    # generated tokens, including the stop token if one was hit
    finish_reason: str   # "stop" or "length"
    queue_ms: float
    ttft_ms: float
    e2e_ms: float


class EngineStream(Protocol):
    """One streaming request (neuroserve.TokenStream or a test fake)."""

    request_id: str

    def read(self, timeout: float | None = None) -> list[int] | None:
        """New token ids; None once the request has finished and every token was read."""

    def result(self) -> EngineOutput: ...


class GenerationEngine(Protocol):
    """What the service needs from an engine (the real neuroserve.Engine, or a test fake).

    run() must be safe to call from many threads at once: the real engine batches concurrent calls.
    """

    def run(self, prompt_ids: list[int], max_new_tokens: int, *, stop_token_ids: list[int],
            client_request_id: str, priority: int) -> EngineOutput: ...

    def stream(self, prompt_ids: list[int], max_new_tokens: int, *, stop_token_ids: list[int],
               client_request_id: str, priority: int) -> EngineStream: ...

    def kv_stats(self) -> dict[str, Any]: ...

    def stats(self) -> dict[str, Any]: ...


class ContextLengthExceeded(ValueError):
    """The prompt is longer than the engine accepts."""


@dataclass(frozen=True)
class GenerationResult:
    text: str              # decoded returned tokens, without the stop token
    token_ids: list[int]   # returned tokens, including the stop token when one was hit
    prompt_tokens: int
    generated_tokens: int  # tokens the engine produced; equals completion_tokens (it stops on the stop token)
    finish_reason: str     # "stop" or "length"
    e2e_ms: float
    ttft_ms: float
    queue_ms: float
    engine_request_id: str

    @property
    def completion_tokens(self) -> int:
        return len(self.token_ids)


class StreamingGeneration:
    """A request whose text is read while it is generated.

    Iterate deltas() (it blocks on the engine, so run it in a worker thread), then call result().
    """

    def __init__(self, service: "EngineService", prompt_ids: list[int], stream: EngineStream) -> None:
        self._service = service
        self._prompt_ids = prompt_ids
        self._stream = stream

    @property
    def engine_request_id(self) -> str:
        return self._stream.request_id

    def deltas(self) -> Iterator[str]:
        """Text as it is generated: one piece per batch of new tokens (never half a character)."""
        detokenizer = self._service.tokenizer.detokenizer()  # skips special tokens such as the stop token
        while (tokens := self._stream.read()) is not None:
            text = "".join(detokenizer.push(token) for token in tokens)
            if text:
                yield text
        tail = detokenizer.flush()
        if tail:
            yield tail

    def result(self) -> GenerationResult:
        """The finished request (raises the engine's error if it failed)."""
        return self._service._result(self._prompt_ids, self._stream.result())


class EngineService:
    """Turns API requests into engine requests.

    No lock: the engine's own loop serialises GPU work and batches requests that arrive together.
    """

    def __init__(self, config: ServerConfig, engine: GenerationEngine, tokenizer: ChatTokenizer,
                 device_name: str) -> None:
        self._config = config
        self._engine = engine
        self._tokenizer = tokenizer
        self._device_name = device_name
        self._stop_token_ids = sorted(tokenizer.stop_token_ids)

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
            max_batch_tokens=e.max_batch_tokens,
            max_pages_per_sequence=e.max_pages_per_sequence,
            device_index=e.device_index,
            scheduling_policy=e.scheduling_policy,
            priority_aging_ms=e.priority_aging_ms,
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

    def engine_stats(self) -> dict[str, Any]:
        return self._engine.stats()

    def chat(self, messages: list[dict[str, Any]], max_new_tokens: int, *,
             ignore_eos: bool = False, priority: int = 1) -> GenerationResult:
        return self._run(self._chat_prompt(messages), max_new_tokens, ignore_eos, priority)

    def complete(self, prompt: str | list[int], max_new_tokens: int, *,
                 ignore_eos: bool = False, priority: int = 1) -> GenerationResult:
        return self._run(self._completion_prompt(prompt), max_new_tokens, ignore_eos, priority)

    def start_chat(self, messages: list[dict[str, Any]], max_new_tokens: int, *,
                   ignore_eos: bool = False, priority: int = 1) -> StreamingGeneration:
        """Like chat(), but returns as soon as the request is queued (bad requests raise here)."""
        return self._start(self._chat_prompt(messages), max_new_tokens, ignore_eos, priority)

    def start_completion(self, prompt: str | list[int], max_new_tokens: int, *,
                         ignore_eos: bool = False, priority: int = 1) -> StreamingGeneration:
        """Like complete(), but returns as soon as the request is queued (bad requests raise here)."""
        return self._start(self._completion_prompt(prompt), max_new_tokens, ignore_eos, priority)

    def _chat_prompt(self, messages: list[dict[str, Any]]) -> list[int]:
        return self._tokenizer.encode_chat(messages, date_string=self._config.model.chat_date_string)

    def _completion_prompt(self, prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, str):
            prompt_ids = self._tokenizer.encode(prompt)  # adds <|begin_of_text|>
        else:
            vocab = self._tokenizer.vocab_size
            bad = next((t for t in prompt if not 0 <= t < vocab), None)
            if bad is not None:
                raise ValueError(f"token id {bad} is outside the vocabulary (0..{vocab - 1})")
            prompt_ids = list(prompt)  # token ids are used verbatim
        return prompt_ids

    def _engine_args(self, prompt_ids: list[int], ignore_eos: bool, priority: int) -> dict[str, Any]:
        limit = self._config.engine.max_prompt_tokens
        if len(prompt_ids) > limit:
            raise ContextLengthExceeded(f"the prompt has {len(prompt_ids)} tokens; the limit is {limit}")
        # The engine stops on the step that produces a stop token (and includes it in the output).
        return {
            "stop_token_ids": [] if ignore_eos else self._stop_token_ids,
            "client_request_id": request_id_var.get() or "",  # links the engine's log line to this request
            "priority": priority,
        }

    def _run(self, prompt_ids: list[int], max_new_tokens: int, ignore_eos: bool, priority: int) -> GenerationResult:
        args = self._engine_args(prompt_ids, ignore_eos, priority)
        return self._result(prompt_ids, self._engine.run(prompt_ids, max_new_tokens, **args))

    def _start(self, prompt_ids: list[int], max_new_tokens: int, ignore_eos: bool,
               priority: int) -> StreamingGeneration:
        args = self._engine_args(prompt_ids, ignore_eos, priority)
        return StreamingGeneration(self, prompt_ids, self._engine.stream(prompt_ids, max_new_tokens, **args))

    def _result(self, prompt_ids: list[int], out: EngineOutput) -> GenerationResult:
        tokens = list(out.tokens)
        stopped = out.finish_reason == "stop"
        return GenerationResult(
            text=self._tokenizer.decode(tokens[:-1] if stopped else tokens),
            token_ids=tokens,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(tokens),
            finish_reason=out.finish_reason,
            e2e_ms=out.e2e_ms,
            ttft_ms=out.ttft_ms,
            queue_ms=out.queue_ms,
            engine_request_id=out.request_id,
        )
