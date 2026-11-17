#!/usr/bin/env python3
"""Release smoke test against a running NeuroServe server (scripts/release_check.sh starts one).

1. The official openai client streams a chat completion, with usage and NeuroServe's timings.
2. Under every scheduling policy, 24 concurrent streamed requests (8 prompts, 3 times each) each
   return exactly the text of the same prompt sent alone, without streaming.

    PYTHONPATH=python python3 scripts/release_smoke.py [--url http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import sys
import threading

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)
    http = httpx.Client(base_url=args.url, timeout=120)
    client = OpenAI(base_url=f"{args.url}/v1", api_key="unused", timeout=120)
    model = http.get("/v1/models").json()["data"][0]["id"]

    def messages(prompt: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def streamed(prompt: str) -> str:
        stream = client.chat.completions.create(model=model, messages=messages(prompt), max_tokens=MAX_TOKENS,
                                                stream=True)
        return "".join(c.choices[0].delta.content or "" for c in stream if c.choices)

    # 1. One streamed request through the official client.
    chunks = list(client.chat.completions.create(model=model, messages=messages(PROMPTS[0]), max_tokens=MAX_TOKENS,
                                                 stream=True, stream_options={"include_usage": True}))
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    usage = chunks[-1].usage
    timing = next((getattr(c, "neuroserve", None) for c in reversed(chunks) if getattr(c, "neuroserve", None)), {})
    print(f"streamed {len(chunks)} chunks: {text!r}")
    print(f"  usage {usage.prompt_tokens} prompt + {usage.completion_tokens} completion tokens; "
          f"server TTFT {timing.get('ttft_ms', float('nan')):.1f} ms, ITL {timing.get('itl_ms', float('nan')):.1f} ms")
    failures = 0 if text and usage.completion_tokens > 0 and timing else 1

    # 2. Concurrent streams under every policy, each compared with the same prompt sent alone.
    expected = []
    for prompt in PROMPTS:
        resp = http.post("/v1/chat/completions", json={"model": model, "messages": messages(prompt),
                                                         "max_tokens": MAX_TOKENS})
        resp.raise_for_status()
        expected.append(resp.json()["choices"][0]["message"]["content"])
    original = http.get("/v1/runtime/policy").json()
    try:
        for policy in original["available"]:
            http.put("/v1/runtime/policy", json={"policy": policy}).raise_for_status()
            jobs = [k % len(PROMPTS) for k in range(CONCURRENT)]
            got: list[str | None] = [None] * CONCURRENT
            errors: list[str] = []

            def worker(k: int) -> None:
                try:
                    got[k] = streamed(PROMPTS[jobs[k]])
                except Exception as exc:  # reported below
                    errors.append(f"{type(exc).__name__}: {exc}")

            threads = [threading.Thread(target=worker, args=(k,)) for k in range(CONCURRENT)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            same = sum(1 for k in range(CONCURRENT) if got[k] == expected[jobs[k]])
            used = http.get("/v1/runtime/policy").json()["policy"]
            note = f"; errors: {errors[:2]}" if errors else ""
            print(f"{policy:<13} {same}/{CONCURRENT} concurrent streams identical to the request alone "
                  f"(policy in use: {used}){note}")
            failures += (CONCURRENT - same) + (used != policy)
    finally:
        http.put("/v1/runtime/policy", json={"policy": original["policy"]})

    latency = http.get("/health").json()["latency"]
    print("server percentiles since start: " + ", ".join(
        f"{name} P50 {latency[name]['p50']:.1f} / P95 {latency[name]['p95']:.1f} ms"
        for name in ("ttft_ms", "itl_ms", "e2e_ms")))
    print("release smoke test " + ("passed" if failures == 0 else f"FAILED ({failures} problems)"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
