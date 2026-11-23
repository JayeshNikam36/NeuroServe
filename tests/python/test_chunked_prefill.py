"""Chunked prefill through the Python API (Day 52): with prefill_chunk_tokens set, a prompt is
prefilled a few tokens per step, sharing steps with other requests' decodes. A step budget smaller
than some prompts shows the point: without chunking those prompts would be refused. Every request
must still match Hugging Face token for token. Its own module, so only one engine is loaded at a time."""

import threading

import pytest

import neuroserve
from test_bindings import MODEL_DIR, needs_model, reference

CHUNK = 4
MAX_BATCH = 4


@pytest.fixture(scope="module")
def refs():
    return [reference(i) for i in range(8)]


@pytest.fixture(scope="module")
def step_budget(refs):
    # one token shorter than the longest reference prompt, which therefore needs chunking
    return max(MAX_BATCH, max(len(p) for p, _ in refs) - 1)


@pytest.fixture(scope="module")
def chunked_engine(step_budget):
    engine = neuroserve.Engine(
        str(MODEL_DIR), max_batch=MAX_BATCH, max_batch_tokens=step_budget, prefill_chunk_tokens=CHUNK
    )
    yield engine
    del engine


@needs_model
def test_chunked_requests_match_hugging_face(chunked_engine, refs, step_budget):
    assert max(len(p) for p, _ in refs) > step_budget, "no prompt is longer than a step: chunking is not needed"
    jobs = [refs[i % 8] for i in range(16)]
    results: list[list[int] | None] = [None] * len(jobs)
    errors: list[Exception] = []

    def worker(k: int) -> None:
        try:
            prompt, expected = jobs[k]
            results[k] = chunked_engine.run(prompt, len(expected)).tokens
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
    stats = chunked_engine.stats()
    assert stats["max_step_tokens"] <= step_budget
    assert stats["mixed_steps"] > 0, "no step mixed prefill chunks and decodes"
    kv = chunked_engine.kv_stats()
    assert kv["free_pages"] == kv["total_pages"], "KV pages leaked"


def test_a_negative_chunk_size_is_refused():
    with pytest.raises(ValueError, match="prefill_chunk_tokens"):
        neuroserve.Engine("/nonexistent", prefill_chunk_tokens=-1)
