"""Open-loop load generator for NeuroServe (and any OpenAI-compatible server).

Sends a seeded schedule (workload.py) at wall-clock times: each request goes out when its time
comes, whether or not earlier ones have finished (open loop, like real traffic). Every request is
streamed, so the client measures its own time to first token and end-to-end time; the server's
own timings come from the neuroserve block of the last chunk.

    python -m neuroserve.bench.loadgen --profile short --seed 1 --out results/short-seed1.json
    python -m neuroserve.bench.loadgen --profile mixed --seed 1 --dry-run   # the schedule only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .workload import DATASET_SHA256, DATASET_VERSION, PROFILES, PlannedRequest, build_schedule, schedule_digest

LOADGEN_VERSION = "1"


@dataclass
class RequestResult:
    index: int
    prompt_id: str
    planned_at_s: float
    sent_at_s: float = 0.0          # when it actually went out (vs planned: the schedule's fidelity)
    status: int = 0                  # HTTP status; 0 = no response
    error: str | None = None         # error code or message, if it failed
    ttft_ms: float | None = None     # client-side: sent -> first text
    e2e_ms: float | None = None      # client-side: sent -> [DONE]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    server: dict[str, Any] = field(default_factory=dict)  # the neuroserve block (server-side timings)

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None


def _text_of(chunk: dict[str, Any]) -> str:
    if not chunk.get("choices"):
        return ""
    choice = chunk["choices"][0]
    return (choice.get("delta") or {}).get("content") or choice.get("text") or ""


async def _send(client: httpx.AsyncClient, planned: PlannedRequest, result: RequestResult, t0: float) -> None:
    start = time.monotonic()
    result.sent_at_s = round(start - t0, 6)
    try:
        async with client.stream("POST", planned.endpoint, json=planned.body) as resp:
            result.status = resp.status_code
            if resp.status_code != 200:
                body = json.loads(await resp.aread() or b"{}")
                result.error = (body.get("error") or {}).get("code") or f"http_{resp.status_code}"
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    result.e2e_ms = (time.monotonic() - start) * 1000.0
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    result.error = chunk["error"].get("code") or "stream_error"
                    return
                if result.ttft_ms is None and _text_of(chunk):
                    result.ttft_ms = (time.monotonic() - start) * 1000.0
                if chunk.get("usage"):
                    result.prompt_tokens = chunk["usage"]["prompt_tokens"]
                    result.completion_tokens = chunk["usage"]["completion_tokens"]
                if chunk.get("neuroserve"):
                    result.server = chunk["neuroserve"]
            if result.e2e_ms is None:
                result.error = "stream_ended_early"
    except httpx.HTTPError as exc:
        result.error = f"{type(exc).__name__}: {exc}"


async def run_schedule(base_url: str, schedule: list[PlannedRequest], *, timeout_s: float = 600.0,
                       transport: httpx.AsyncBaseTransport | None = None) -> tuple[list[RequestResult], float]:
    """Sends every request at its planned time; returns the results and the run's duration (s)."""
    results = [RequestResult(p.index, p.prompt_id, p.at_s) for p in schedule]
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)  # never queue in the client
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s, limits=limits, transport=transport) as client:
        t0 = time.monotonic()
        tasks = []
        for planned, result in zip(schedule, results):
            delay = t0 + planned.at_s - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(_send(client, planned, result, t0)))
        await asyncio.gather(*tasks)
        duration = time.monotonic() - t0
    return results, duration


def percentile(values: list[float], q: float) -> float | None:
    """Exact nearest-rank percentile (a run has at most a few thousand requests)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q * len(ordered))) - 1]


def summarize(results: list[RequestResult], duration_s: float) -> dict[str, Any]:
    ok = [r for r in results if r.ok]

    def stats(values: list[float]) -> dict[str, Any]:
        return {"count": len(values),
                "mean": round(sum(values) / len(values), 3) if values else None,
                **{f"p{int(q * 100)}": (round(v, 3) if (v := percentile(values, q)) is not None else None)
                   for q in (0.5, 0.95, 0.99)}}

    output_tokens = sum(r.completion_tokens or 0 for r in ok)
    errors: dict[str, int] = {}
    for r in results:
        if not r.ok:
            errors[r.error or f"http_{r.status}"] = errors.get(r.error or f"http_{r.status}", 0) + 1
    return {
        "requests": len(results),
        "succeeded": len(ok),
        "errors": errors,
        "duration_s": round(duration_s, 3),
        "throughput_rps": round(len(ok) / duration_s, 3) if duration_s > 0 else None,
        "output_tokens_per_s": round(output_tokens / duration_s, 1) if duration_s > 0 else None,
        "max_send_lag_ms": round(max((r.sent_at_s - r.planned_at_s) * 1000.0 for r in results), 3),
        "client_ttft_ms": stats([r.ttft_ms for r in ok if r.ttft_ms is not None]),
        "client_e2e_ms": stats([r.e2e_ms for r in ok if r.e2e_ms is not None]),
        "server_queue_ms": stats([r.server["queue_ms"] for r in ok if "queue_ms" in r.server]),
        "server_itl_ms": stats([r.server["itl_ms"] for r in ok if "itl_ms" in r.server]),
    }


def _get_json(base_url: str, path: str) -> dict[str, Any]:
    try:
        return httpx.get(f"{base_url}{path}", timeout=10).json()
    except (httpx.HTTPError, ValueError):
        return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neuroserve.bench.loadgen", description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="server base URL")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="short")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--requests", type=int, help="override the profile's number of requests")
    parser.add_argument("--rate", type=float, help="override the profile's arrival rate (requests/s)")
    parser.add_argument("--model", help="model name to send (default: the server's first model)")
    parser.add_argument("--out", type=Path, help="write the full results as JSON here")
    parser.add_argument("--dry-run", action="store_true", help="print the schedule's fingerprint; send nothing")
    args = parser.parse_args(argv)

    profile = PROFILES[args.profile]
    model = args.model
    if model is None:
        models = _get_json(args.url, "/v1/models").get("data") if not args.dry_run else None
        model = models[0]["id"] if models else "llama-3.2-1b-instruct"
    schedule = build_schedule(profile, args.seed, model, num_requests=args.requests, rate_rps=args.rate)
    digest = schedule_digest(schedule)
    print(f"profile={profile.name} seed={args.seed} requests={len(schedule)} "
          f"last_arrival={schedule[-1].at_s:.3f}s schedule={digest[:16]}")
    if args.dry_run:
        return 0

    health = _get_json(args.url, "/health")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results, duration = asyncio.run(run_schedule(args.url, schedule))
    summary = summarize(results, duration)
    report = {
        "meta": {"loadgen_version": LOADGEN_VERSION, "started_at": started, "url": args.url, "model": model,
                 "profile": asdict(profile), "seed": args.seed, "requests": len(schedule),
                 "rate_rps": args.rate or profile.rate_rps, "schedule_sha256": digest,
                 "dataset": {"version": DATASET_VERSION, "sha256": DATASET_SHA256},
                 "server": {"policy": (health.get("engine") or {}).get("policy"), "gpu": health.get("gpu"),
                            "model": health.get("model")}},
        "summary": summary,
        "requests": [asdict(r) for r in results],
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if summary["succeeded"] == len(schedule) else 1


if __name__ == "__main__":
    sys.exit(main())
