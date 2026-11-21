"""Preemption by recompute with prefix caching on (Day 38): a small KV cache forces preemptions; a
preempted request resumes from its prompt and the tokens it already generated, reusing its own
CACHED pages where they survived. Every request must still match Hugging Face token for token.
A separate module from test_preemption.py, so the two engines are never loaded at the same time."""

import math
import threading

import pytest

import neuroserve
from test_bindings import MODEL_DIR, needs_model, reference

PAGE_SIZE = 16


@pytest.fixture(scope="module")
def refs():
    return [reference(i) for i in range(8)]


@pytest.fixture(scope="module")
def cached_engine(refs):
    largest = max(math.ceil((len(p) + len(e)) / PAGE_SIZE) for p, e in refs)
    engine = neuroserve.Engine(str(MODEL_DIR), num_pages=largest + 1, max_batch=8, prefix_caching=True)
    yield engine
    del engine


@needs_model
@pytest.mark.parametrize("policy", neuroserve.scheduling_policies)
def test_preempted_requests_resume_with_the_cache_and_match_hugging_face(cached_engine, refs, policy):
    cached_engine.set_scheduling_policy(policy)
    before = cached_engine.stats()
    jobs = [refs[i % 8] for i in range(24)]
    results: list[list[int] | None] = [None] * len(jobs)
    errors: list[Exception] = []

    def worker(k: int) -> None:
        try:
            prompt, expected = jobs[k]
            results[k] = cached_engine.run(prompt, len(expected), priority=k % 3).tokens
        except Exception as exc:  # collected and reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(len(jobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)

    assert not errors, errors[:3]
    mismatched = [k for k, (_, expected) in enumerate(jobs) if results[k] != expected]
    assert not mismatched, f"{len(mismatched)} of {len(jobs)} differ from Hugging Face (first: request {mismatched[0]})"
    stats = cached_engine.stats()
    assert stats["preemptions"] > before["preemptions"], "the KV cache was never short: preemption was not exercised"
    assert stats["completed"] - before["completed"] == len(jobs)
    kv = cached_engine.kv_stats()
    assert kv["free_pages"] == kv["total_pages"], "KV pages leaked"
    assert kv["prefix_lookups"] > 0
