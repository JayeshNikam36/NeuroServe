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
    kv = engine.kv_stats()
    assert kv["free_pages"] == kv["total_pages"], "KV pages leaked after errors"


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