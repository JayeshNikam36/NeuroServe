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
    itl_ms: float        # mean time between consecutive tokens
    max_itl_ms: float


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

    def memory(self) -> dict[str, Any]: ...

    def stats(self) -> dict[str, Any]: ...

    def latency(self) -> dict[str, Any]: ...

    def decisions(self, since: int = 0, limit: int = 100) -> dict[str, Any]: ...

    def step_profile(self, records: bool = False) -> dict[str, Any]: ...

    def reset_step_profile(self) -> None: ...

    def cancel(self, request_id: str) -> bool: ...

    def set_scheduling_policy(self, name: str) -> None:
        """From the next engine step on; an unknown name raises ValueError."""


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
    itl_ms: float          # mean inter-token latency
    max_itl_ms: float      # the longest gap between two tokens
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

    def cancel(self) -> bool:
        """Stops the request at the next engine step (False if it has already finished)."""
        return self._service.cancel(self.engine_request_id)

    def deltas(self, poll_s: float = 0.1) -> Iterator[str]:
        """Text as it is generated: one piece per batch of new tokens (never half a character).

        Yields "" when nothing new arrived within poll_s seconds, so a caller iterating this in a
        worker thread gets control back regularly (e.g. to notice that its client has gone).
        """
        detokenizer = self._service.tokenizer.detokenizer()  # skips special tokens such as the stop token
        while (tokens := self._stream.read(timeout=poll_s)) is not None:
            yield "".join(detokenizer.push(token) for token in tokens)
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
        self._requested_policy: str | None = None  # the last set_scheduling_policy() name

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
            prefill_chunk_tokens=e.prefill_chunk_tokens,
            mixed_steps=e.mixed_steps,
            cuda_graphs=e.cuda_graphs,
            max_pages_per_sequence=e.max_pages_per_sequence,
            device_index=e.device_index,
            scheduling_policy=e.scheduling_policy,
            priority_aging_ms=e.priority_aging_ms,
            max_queue=e.max_queue,
            queue_timeout_ms=e.queue_timeout_ms,
            decision_log_capacity=e.decision_log_capacity,
            prefix_caching=e.prefix_caching,
            kv_admission=e.kv_admission,
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

    def latency(self) -> dict[str, Any]:
        return self._engine.latency()

    def runtime_state(self) -> dict[str, Any]:
        """The KV cache's state and counters, for GET /v1/runtime/state (docs/kv-cache.md, Day 40).

        Occupancy counts pages held by live requests. CACHED pages (released, still reusable) are
        reclaimed on demand, so they are neither used nor plainly free; both are shown. Prefix
        counters cover full pages: a lookup could reuse lookup_pages, it reused hit_pages and
        missed the rest; reused_tokens are the prompt tokens that were not prefilled.
        """
        kv = self._engine.kv_stats()
        stats = self._engine.stats()
        total = kv["total_pages"]
        reclaimable = kv["free_pages"]  # FREE + CACHED
        used = total - reclaimable
        lookup = kv["prefix_lookup_pages"]
        hits = kv["prefix_hit_pages"]
        return {
            "kv_cache": {
                "page_size_tokens": kv["page_size_tokens"],
                "total_pages": total,
                "used_pages": used,
                "cached_pages": kv["cached_pages"],
                "free_pages": reclaimable - kv["cached_pages"],
                "occupancy": round(used / total, 4) if total else 0.0,
                "shared_pages": kv["shared_pages"],
                "indexed_pages": kv["indexed_pages"],
                "evicted_pages": kv["evicted_pages"],
            },
            "prefix_cache": {
                "enabled": self._config.engine.prefix_caching,
                "lookups": kv["prefix_lookups"],
                "lookup_pages": lookup,
                "hit_pages": hits,
                "miss_pages": lookup - hits,
                "hit_rate": round(hits / lookup, 4) if lookup else 0.0,
                "reused_tokens": kv["prefix_reused_tokens"],
            },
            "admission": self._config.engine.kv_admission,
            "preemptions": stats["preemptions"],
            "policy": stats["policy"],
            "gpu_memory": self.memory_summary(),
        }

    def memory_summary(self) -> dict[str, Any]:
        """GPU memory by owner, in MiB (docs/gpu-memory.md, Day 58): each owner's bytes through the
        pool and recorded from the engine, the pool, and the device's used memory beside them."""
        m = self._engine.memory()

        def mib(n: int) -> float:
            return round(n / 1048576, 1)

        return {
            "owners": {name: {"mib": mib(u["bytes"]), "pool_mib": mib(u["pool_bytes"]),
                              "external_mib": mib(u["external_bytes"]), "peak_mib": mib(u["peak_bytes"]),
                              "allocations": u["allocations"]}
                       for name, u in m["owners"].items()},
            "items": [{"owner": i["owner"], "source": i["source"], "label": i["label"], "mib": mib(i["bytes"]),
                       "allocations": i["allocations"]} for i in m["items"]],
            "attributed_mib": mib(m["bytes"]),
            "pool": {"supported": m["pool"]["supported"],
                     "release_threshold_mib": mib(m["pool"]["release_threshold_bytes"]),
                     "reserved_mib": mib(m["pool"]["reserved_bytes"]), "used_mib": mib(m["pool"]["used_bytes"])},
            "device": {"total_mib": mib(m["device"]["total_bytes"]), "used_mib": mib(m["device"]["used_bytes"]),
                       "unattributed_mib": mib(m["device"]["unattributed_bytes"])},
        }

    def decisions(self, since: int, limit: int) -> dict[str, Any]:
        return self._engine.decisions(since=since, limit=limit)

    def step_profile(self, records: bool) -> dict[str, Any]:
        """Where the engine thread's time went since the last reset (docs/profiling.md)."""
        return self._engine.step_profile(records=records)

    def reset_step_profile(self) -> None:
        self._engine.reset_step_profile()

    def cancel(self, engine_request_id: str) -> bool:
        return self._engine.cancel(engine_request_id)

    def scheduling_policy(self) -> dict[str, Any]:
        """The policy deciding now, and a requested one that is not applied yet ("pending").

        The engine switches policy at its next step; an idle engine takes no steps, so a policy
        requested while idle stays pending until the next request arrives.
        """
        current = self._engine.stats()["policy"]
        pending = self._requested_policy if self._requested_policy not in (None, current) else None
        return {"policy": current, "pending": pending}

    def set_scheduling_policy(self, name: str) -> dict[str, Any]:
        """Requests a policy for the next engine step on (unknown name: ValueError)."""
        self._engine.set_scheduling_policy(name)
        self._requested_policy = name
        return self.scheduling_policy()

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
            itl_ms=out.itl_ms,
            max_itl_ms=out.max_itl_ms,
            engine_request_id=out.request_id,
        )
