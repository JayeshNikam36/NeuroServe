"""Preemption on the real GPU: a deliberately small KV cache forces the engine to preempt requests,
and every request must still match Hugging Face token for token (run inside the dev container)."""

import math
import threading

import pytest

import neuroserve
from test_bindings import MODEL_DIR, needs_model, reference

PAGE_SIZE = 16  # fixed by the engine's decode kernel


@pytest.fixture(scope="module")
def refs():
    return [reference(i) for i in range(8)]


@pytest.fixture(scope="module")
def small_engine(refs):
    # Room for the largest request plus one page: every request fits alone (so it can always
    # finish), but two or more concurrent requests cannot all grow to full length.
    largest = max(math.ceil((len(p) + len(e)) / PAGE_SIZE) for p, e in refs)
    return neuroserve.Engine(str(MODEL_DIR), num_pages=largest + 1, max_batch=8)


@needs_model
@pytest.mark.parametrize("policy", ["fcfs", "token-budget", "decode-first"])
def test_preempted_requests_restart_and_still_match_hugging_face(small_engine, refs, policy):
    # Every policy must produce exactly the Hugging Face tokens; only the order of the work differs.
    small_engine.set_scheduling_policy(policy)
    before = small_engine.stats()
    jobs = [refs[i % 8] for i in range(24)]
    results: list[list[int] | None] = [None] * len(jobs)
    errors: list[Exception] = []

    def worker(k: int) -> None:
        try:
            prompt, expected = jobs[k]
            results[k] = small_engine.generate(prompt, max_new_tokens=len(expected))
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
    stats = small_engine.stats()
    assert stats["policy"] == policy
    assert stats["preemptions"] > before["preemptions"], "the KV cache was never short: preemption was not exercised"
    assert stats["completed"] - before["completed"] == len(jobs)
    kv = small_engine.kv_stats()
    assert kv["free_pages"] == kv["total_pages"], "KV pages leaked"
