"""Tests for the neuroserve Python bindings (run inside the dev container)."""

import os
import threading
from array import array
from pathlib import Path

import pytest

import neuroserve

MODEL_DIR = Path(os.environ.get("NS_MODEL_DIR", "/models/llama-3.2-1b-fp16"))
REF_DIR = Path(os.environ.get("NS_REF_DIR", "/models/llama-3.2-1b-ref"))
HAVE_MODEL = (MODEL_DIR / "weights.bin").exists() and (REF_DIR / "prompt_00_ids.i32").exists()
needs_model = pytest.mark.skipif(
    not HAVE_MODEL, reason=f"model/reference files not found ({MODEL_DIR}, {REF_DIR})"
)


def read_i32(path: Path) -> list[int]:
    values = array("i")
    values.frombytes(path.read_bytes())
    return values.tolist()


def reference(i: int) -> tuple[list[int], list[int]]:
    return (
        read_i32(REF_DIR / f"prompt_{i:02d}_ids.i32"),
        read_i32(REF_DIR / f"prompt_{i:02d}_greedy.i32"),
    )


@pytest.fixture(scope="module")
def engine():
    return neuroserve.Engine(str(MODEL_DIR), num_pages=512)


def test_version_is_semver():
    assert neuroserve.__version__.count(".") == 2


def test_config_offers_exactly_the_engine_policies():
    import typing

    from neuroserve.server.config import EngineSection

    allowed = typing.get_args(EngineSection.model_fields["scheduling_policy"].annotation)
    assert set(allowed) == set(neuroserve.scheduling_policies) == {"fcfs", "token-budget", "decode-first", "priority"}


def test_unknown_scheduling_policy_is_rejected_before_loading():
    with pytest.raises(ValueError, match="unknown scheduling policy 'lifo'"):
        neuroserve.Engine("/no/such/model", scheduling_policy="lifo")


def test_device_info_reports_gpu():
    info = neuroserve.device_info()
    assert info["name"]
    assert int(info["compute_capability"].split(".")[0]) >= 8
    assert info["total_mem_bytes"] > info["free_mem_bytes"] > 0


@needs_model
def test_generate_matches_hugging_face(engine):
    for i in range(8):
        prompt, expected = reference(i)
        assert engine.generate(prompt, max_new_tokens=len(expected)) == expected, f"prompt {i}"
    kv = engine.kv_stats()
    assert kv["free_pages"] == kv["total_pages"], "KV pages leaked"


@needs_model
def test_bad_requests_raise_python_exceptions(engine):
    prompt, _ = reference(0)
    with pytest.raises(ValueError):
        engine.generate([], max_new_tokens=4)            # empty prompt
    with pytest.raises(ValueError):
        engine.generate([1000] * 600, max_new_tokens=4)  # longer than max_prompt_tokens
    with pytest.raises(ValueError):
        engine.generate(prompt, max_new_tokens=0)        # nothing to generate
    with pytest.raises(ValueError):
        engine.run(prompt, 4, priority=3)                # priority is 0, 1 or 2
    kv = engine.kv_stats()
    assert kv["free_pages"] == kv["total_pages"], "KV pages leaked after errors"


@needs_model
def test_stream_returns_the_tokens_of_generate_as_they_are_made(engine):
    for i in range(8):
        prompt, expected = reference(i)
        stream = engine.stream(prompt, max_new_tokens=len(expected))
        tokens, reads = [], 0
        while (batch := stream.read()) is not None:
            tokens += batch
            reads += 1
        assert tokens == expected == stream.result().tokens, f"prompt {i}"
        assert reads > 1, "all tokens arrived in one read: nothing was streamed"
    kv = engine.kv_stats()
    assert kv["free_pages"] == kv["total_pages"], "KV pages leaked"


@needs_model
def test_generate_releases_the_gil(engine):
    """A Python thread must keep running while generate() works in C++."""
    prompt, expected = reference(0)
    ticks = 0
    stop = threading.Event()

    def spin():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1

    worker = threading.Thread(target=spin)
    worker.start()
    before = ticks
    engine.generate(prompt, max_new_tokens=len(expected))
    during = ticks - before
    stop.set()
    worker.join()
    assert during > 10_000, f"Python thread barely ran during generate ({during} ticks): GIL not released?"

# ---- Day 12B: the engine loop behind Engine.run() -------------------------------------------

@needs_model
def test_run_stops_on_the_stop_token(engine):
    prompt, expected = reference(0)
    stop = expected[5]
    first = expected.index(stop)  # the first occurrence ends generation
    result = engine.run(prompt, len(expected), stop_token_ids=[stop], client_request_id="test")
    assert result.tokens == expected[: first + 1]  # includes the stop token
    assert result.finish_reason == "stop"
    assert result.request_id.startswith("ns-")
    assert 0 <= result.ttft_ms <= result.e2e_ms


@needs_model
def test_100_concurrent_requests_are_batched_and_match_hugging_face(engine):
    jobs = [reference(i % 8) for i in range(100)]
    results: list[list[int] | None] = [None] * len(jobs)
    errors: list[Exception] = []

    def worker(k: int) -> None:
        try:
            prompt, expected = jobs[k]
            results[k] = engine.generate(prompt, max_new_tokens=len(expected))
        except Exception as exc:  # collected and reported below, so a thread error cannot pass silently
            errors.append(exc)

    before = engine.stats()
    threads = [threading.Thread(target=worker, args=(k,)) for k in range(len(jobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)

    assert not errors, errors[:3]
    mismatched = [k for k, (_, expected) in enumerate(jobs) if results[k] != expected]
    assert not mismatched, (f"{len(mismatched)} of 100 differ from Hugging Face "
                            f"(first: request {mismatched[0]}, reference prompt {mismatched[0] % 8})")
    after = engine.stats()
    assert after["completed"] - before["completed"] == 100
    assert after["max_batch_seen"] > 1, "concurrent requests were not batched"
    kv = engine.kv_stats()
    assert kv["free_pages"] == kv["total_pages"], "KV pages leaked"
