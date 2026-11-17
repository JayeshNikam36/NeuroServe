#!/usr/bin/env python3
"""Release smoke test against a running NeuroServe server (scripts/release_check.sh starts one).

What NeuroServe guarantees, and so what fails this test (see ADR-0006):
1. The official openai client streams a chat completion, with usage and NeuroServe's timings.
2. Streaming changes nothing: each prompt streamed alone returns the same tokens and text as the
   same prompt sent alone without streaming.
3. Under every scheduling policy, 24 concurrent streams (8 prompts, 3 times each) all complete
   correctly, and 8 identical requests sent at once return identical tokens (determinism).

Reported but not failed: how many concurrent streams match the same prompt sent alone. Decoding
is deterministic but not batch-invariant, so at a near-tie the batch can change a greedy choice.

    PYTHONPATH=python python3 scripts/release_smoke.py [--url http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Callable
from typing import Any

import httpx
from openai import OpenAI

PROMPTS = [
    "What is the capital of France?",
    "Explain in one sentence what a GPU is.",
    "Name three primary colors.",
    "Write one sentence about the ocean.",
    "What is 17 multiplied by 6?",
    "Give one tip for writing clear emails.",
    "Why is the sky blue? Answer briefly.",
    "Suggest a name for a friendly robot.",
]
CONCURRENT = 24
MAX_TOKENS = 48


def first_difference(a: list[int], b: list[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)
    http = httpx.Client(base_url=args.url, timeout=300)
    model = http.get("/v1/models").json()["data"][0]["id"]
    problems: list[str] = []

    def body(prompt: str, stream: bool) -> dict[str, Any]:
        return {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS,
                "stream": stream, "stream_options": {"include_usage": True} if stream else None,
                "neuroserve": {"return_token_ids": True}}

    def alone(prompt: str) -> dict[str, Any]:
        resp = http.post("/v1/chat/completions", json={k: v for k, v in body(prompt, False).items() if v is not None})
        resp.raise_for_status()
        data = resp.json()
        return {"text": data["choices"][0]["message"]["content"], "ids": data["neuroserve"]["completion_token_ids"],
                "finish": data["choices"][0]["finish_reason"], "usage": data["usage"]["completion_tokens"]}

    def streamed(prompt: str) -> dict[str, Any]:
        out: dict[str, Any] = {"text": "", "ids": None, "finish": None, "usage": None, "done": False}
        with http.stream("POST", "/v1/chat/completions", json=body(prompt, True)) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line == "data: [DONE]":
                    out["done"] = True
                elif line.startswith("data: "):
                    chunk = json.loads(line[6:])
                    if "error" in chunk:
                        raise RuntimeError(chunk["error"])
                    for choice in chunk.get("choices") or []:
                        out["text"] += choice["delta"].get("content") or ""
                        out["finish"] = choice.get("finish_reason") or out["finish"]
                    if chunk.get("neuroserve"):
                        out["ids"] = chunk["neuroserve"]["completion_token_ids"]
                    if chunk.get("usage"):
                        out["usage"] = chunk["usage"]["completion_tokens"]
        return out

    def complete(r: dict[str, Any]) -> bool:
        return bool(r.get("done", True) and r["finish"] and r["ids"] and r["usage"] == len(r["ids"]) and r["text"])

    def at_once(fn: Callable[[str], dict[str, Any]], prompts: list[str]) -> tuple[list[Any], list[str]]:
        results: list[Any] = [None] * len(prompts)
        errors: list[str] = []

        def worker(k: int) -> None:
            try:
                results[k] = fn(prompts[k])
            except Exception as exc:  # reported below
                errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(len(prompts))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results, errors

    # 1. The official openai client.
    client = OpenAI(base_url=f"{args.url}/v1", api_key="unused", timeout=120)
    chunks = list(client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": PROMPTS[0]}], max_tokens=MAX_TOKENS, stream=True,
        stream_options={"include_usage": True}))
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    usage = chunks[-1].usage
    timing = next((getattr(c, "neuroserve", None) for c in reversed(chunks) if getattr(c, "neuroserve", None)), {})
    print(f"1. openai client streamed {len(chunks)} chunks: {text!r}")
    print(f"   usage {usage.prompt_tokens} prompt + {usage.completion_tokens} completion tokens; "
          f"server TTFT {timing.get('ttft_ms', float('nan')):.1f} ms, ITL {timing.get('itl_ms', float('nan')):.1f} ms")
    if not (text and usage.completion_tokens > 0 and timing):
        problems.append("the openai client stream is incomplete")

    # 2. Streaming changes nothing.
    reference = [alone(p) for p in PROMPTS]
    same = 0
    for p, ref in zip(PROMPTS, reference):
        s = streamed(p)
        same += s["ids"] == ref["ids"] and s["text"] == ref["text"] and complete(s)
    print(f"2. streamed alone vs not streamed: {same}/{len(PROMPTS)} identical (tokens and text)")
    if same != len(PROMPTS):
        problems.append("streaming changed a response")

    # 3. Every policy: concurrent streams complete; identical requests agree (determinism).
    original = http.get("/v1/runtime/policy").json()
    try:
        for policy in original["available"]:
            http.put("/v1/runtime/policy", json={"policy": policy}).raise_for_status()
            jobs = [k % len(PROMPTS) for k in range(CONCURRENT)]
            results, errors = at_once(streamed, [PROMPTS[j] for j in jobs])
            ok = [r for r in results if r is not None and complete(r)]
            diverged = sorted({(f"p{j}", first_difference(r["ids"], reference[j]["ids"]))
                               for j, r in zip(jobs, results) if r is not None and r["ids"] != reference[j]["ids"]})
            # The prompt seen to change at a near-tie in a larger batch: identical copies must still agree.
            copies, copy_errors = at_once(alone, [PROMPTS[-1]] * 8)
            distinct = len({tuple(c["ids"]) for c in copies if c is not None})
            used = http.get("/v1/runtime/policy").json()["policy"]
            matching = CONCURRENT - sum(1 for j, r in zip(jobs, results) if r is None or r["ids"] != reference[j]["ids"])
            note = "" if not diverged else " (first difference: " + ", ".join(f"{p} at token {i}" for p, i in diverged) + ")"
            print(f"3. {policy:<13} {len(ok)}/{CONCURRENT} streams complete; 8 identical requests -> "
                  f"{distinct} distinct output(s); {matching}/{CONCURRENT} match the prompt alone{note}")
            for e in (errors + copy_errors)[:2]:
                print(f"   error: {e}")
            if len(ok) != CONCURRENT or errors or copy_errors:
                problems.append(f"{policy}: {CONCURRENT - len(ok)} streams failed or were incomplete")
            if distinct != 1:
                problems.append(f"{policy}: identical requests sent at once returned {distinct} different outputs")
            if used != policy:
                problems.append(f"{policy}: the server used {used}")
    finally:
        http.put("/v1/runtime/policy", json={"policy": original["policy"]})

    latency = http.get("/health").json()["latency"]
    print("server percentiles since start: " + ", ".join(
        f"{name} P50 {latency[name]['p50']:.1f} / P95 {latency[name]['p95']:.1f} ms"
        for name in ("ttft_ms", "itl_ms", "e2e_ms")))
    for p in problems:
        print(f"PROBLEM: {p}")
    print("release smoke test " + ("passed" if not problems else f"FAILED ({len(problems)} problems)"))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
