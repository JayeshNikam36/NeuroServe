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


@pytest.mark.parametrize("options", [{"max_queue": 0}, {"queue_timeout_ms": -1}, {"decision_log_capacity": 0}])
def test_bad_queue_limits_are_rejected_before_loading(options):
    with pytest.raises(ValueError):
        neuroserve.Engine("/no/such/model", **options)


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
def test_cancel_stops_a_running_request_and_frees_its_pages(engine):
    prompt, _ = reference(0)
    stream = engine.stream(prompt, max_new_tokens=200)
    tokens = stream.read()
    assert tokens, "the request should be running"
    assert engine.cancel(stream.request_id) is True
    while (batch := stream.read()) is not None:
        tokens += batch
    assert len(tokens) < 200, "the request ran to the end: it was not cancelled"
    with pytest.raises(neuroserve.RequestCancelled):
        stream.result()
    assert engine.cancel(stream.request_id) is False  # finished: nothing left to cancel
    assert engine.cancel("ns-does-not-exist") is False
    kv = engine.kv_stats()
    assert kv["free_pages"] == kv["total_pages"], "KV pages leaked after a cancellation"


@needs_model
def test_latency_is_recorded_per_request_and_as_percentiles(engine):
    before = engine.latency()
    results = [engine.run(prompt, len(expected)) for prompt, expected in map(reference, range(4))]
    for r in results:
        n = len(r.tokens)
        assert n > 1 and 0 < r.itl_ms <= r.max_itl_ms
        # The gaps between tokens add up to the time from the first token to the end.
        assert r.itl_ms * (n - 1) <= r.e2e_ms - r.ttft_ms + 0.01
        assert r.itl_ms * (n - 1) >= 0.9 * (r.e2e_ms - r.ttft_ms) - 1.0
    after = engine.latency()
    assert after["ttft_ms"]["count"] - before["ttft_ms"]["count"] == 4
    assert after["itl_ms"]["count"] - before["itl_ms"]["count"] == sum(len(r.tokens) - 1 for r in results)
    for metric in ("queue_ms", "ttft_ms", "itl_ms", "e2e_ms"):
        s = after[metric]
        assert 0 <= s["p50"] <= s["p95"] <= s["p99"] <= s["max"], metric


@needs_model
def test_every_admission_is_explained(engine):
    cursor = 0
    while (page := engine.decisions(since=cursor, limit=1000))["data"]:  # skip what earlier tests logged
        cursor = page["next_since"]
    ids = [engine.run(prompt, len(expected)).request_id for prompt, expected in map(reference, range(2))]
    admits = {d["request_id"]: d for d in engine.decisions(since=cursor, limit=1000)["data"]
              if d["action"] == "admit"}
    for request_id in ids:
        d = admits[request_id]
        assert d["reason"] == "admitted" and d["policy"] == engine.stats()["policy"]
        assert d["numbers"]["pages_needed"] <= d["numbers"]["free_pages"]
    assert engine.decisions(since=10**18)["data"] == []  # nothing newer than a cursor far ahead


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


@needs_model
def test_identical_requests_in_one_batch_produce_identical_tokens(engine):
    # Guaranteed (ADR-0006): determinism. The same work in the same batch gives the same tokens.
    # Not guaranteed: the same tokens in every batch (a near-tie can depend on the batch size).
    prompt, expected = reference(3)
    results: list[list[int] | None] = [None] * 8

    def worker(k: int) -> None:
        results[k] = engine.generate(prompt, max_new_tokens=len(expected))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    assert all(r is not None for r in results)
    assert len({tuple(r) for r in results}) == 1, "identical concurrent requests returned different tokens"
