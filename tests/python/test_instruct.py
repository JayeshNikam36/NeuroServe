"""Llama-3.2-1B-Instruct served by NeuroServe must match Hugging Face greedy output on chat prompts."""

import json
import os
from pathlib import Path

import pytest

import neuroserve

INSTRUCT_DIR = Path(os.environ.get("NS_INSTRUCT_MODEL_DIR", "/models/llama-3.2-1b-instruct-fp16"))
GREEDY_REF = Path(os.environ.get("NS_INSTRUCT_REF", "/models/llama-3.2-1b-ref/instruct_greedy.json"))

pytestmark = pytest.mark.skipif(
    not ((INSTRUCT_DIR / "weights.bin").exists() and GREEDY_REF.exists()),
    reason=f"instruct weights or reference not found ({INSTRUCT_DIR}, {GREEDY_REF})",
)


def first_difference(a: list[int], b: list[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def test_instruct_chat_prompts_match_hugging_face():
    ref = json.loads(GREEDY_REF.read_text(encoding="utf-8"))
    assert len(ref["cases"]) == 5
    engine = neuroserve.Engine(str(INSTRUCT_DIR), num_pages=512)
    try:
        for case in ref["cases"]:
            expected = case["greedy_ids"]
            out = engine.generate(case["prompt_ids"], max_new_tokens=len(expected))
            assert out == expected, (
                f"{case['name']}: first difference at generated token {first_difference(out, expected)} "
                f"of {len(expected)}"
            )
        kv = engine.kv_stats()
        assert kv["free_pages"] == kv["total_pages"]
    finally:
        del engine  # release GPU memory before later tests load another model
