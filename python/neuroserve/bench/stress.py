"""Stress run: sustained high-concurrency mixed load, sampled every 10 s (see docs/benchmarking.md).

    python -m neuroserve.bench.stress --config configs/dev.yaml --out-dir benchmarks/results/raw/stress-1

Starts a server from the base config (prefix caching on, a quarter of the KV pool, so eviction and
preemption are exercised), then keeps --concurrency requests in flight for --duration-min minutes:
each client sends its next request as soon as its last one ends, drawn from the mixed,
shared_prefix and memory_pressure profiles, and a share of them is cancelled mid-stream. Every
--interval-s it records the KV pages, the engine's queue, GPU memory (NVML), the server's resident
memory, and the latency of the requests that finished in that interval. At the end it drains the
load and checks: no failed requests, the engine still serving and its invariant checks never
tripped, no KV pages held after the drain, flat GPU and host memory, and latency and throughput at
the end of the run as at its start. Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import random
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import httpx
import yaml

from neuroserve.server.config import ServerConfig

from . import compare, loadgen, prefix_ab
from .workload import PROFILES, PlannedRequest, build_schedule

MIX = (("mixed", 0.5), ("shared_prefix", 0.3), ("memory_pressure", 0.2))
PROBE = PROFILES["short"]  # one small request after the drain: is the engine still serving?
LOG_FAILURE_MARKERS = ("violat", "Traceback")  # an invariant violation, an unhandled exception


def request_source(mix: tuple[tuple[str, float], ...], seed: int, model: str) -> Iterator[tuple[str, PlannedRequest]]:
    """An endless seeded stream of (profile, request): each pick follows the mix's weights; each
    profile's requests come from its own schedules, a new seed every 200 requests (so new prefixes
    keep arriving and old ones age out of the cache)."""
    picker = random.Random(f"neuroserve:stress:{seed}")
    streams = {name: (p for s in itertools.count(seed * 1000)
                      for p in build_schedule(PROFILES[name], s, model, num_requests=200))
               for name, _ in mix}
    names, weights = [n for n, _ in mix], [w for _, w in mix]
    while True:
        name = picker.choices(names, weights=weights)[0]
        yield name, next(streams[name])


def window_metrics(done: list[tuple[float, loadgen.RequestResult]], start_s: float, end_s: float) -> dict[str, Any]:
    """Latency and throughput of the requests that ended in [start_s, end_s): client-cancelled ones
    count only as cancelled."""
    mine = [r for t, r in done if start_s <= t < end_s]
    ok = [r for r in mine if r.ok]
    ttft = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    itl = [r.server["itl_ms"] for r in ok if "itl_ms" in r.server]
    span = end_s - start_s

    def pct(values: list[float], q: float) -> float | None:
        v = loadgen.percentile(values, q)
        return round(v, 2) if v is not None else None

    return {
        "finished": len(ok), "failed": sum(1 for r in mine if not r.ok and r.error != "cancelled_by_client"),
        "cancelled": sum(1 for r in mine if r.error == "cancelled_by_client"),
        "ttft_p50_ms": pct(ttft, 0.5), "ttft_p95_ms": pct(ttft, 0.95), "itl_p50_ms": pct(itl, 0.5),
        "output_tokens_per_s": round(sum(r.completion_tokens or 0 for r in ok) / span, 1) if span > 0 else None,
    }


def invariant_checks() -> bool | None:
    """Whether this build's engines check the KV invariants after every step; None if it does not
    say (a build from before Day 44)."""
    try:
        from neuroserve import _core
    except ImportError:
        return None
    return getattr(_core, "check_invariants", None)


def rss_mib(pid: int) -> float | None:
    """A process's resident memory (Linux /proc), or None."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


class GpuMemory:
    """Device memory in use (NVML), or None without NVML."""

    def __init__(self, index: int) -> None:
        self._nvml = self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml, self._handle = pynvml, pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:  # noqa: BLE001  (no NVML, no driver, no GPU: memory is simply not sampled)
            self._nvml = None

    def used_mib(self) -> float | None:
        if self._nvml is None:
            return None
        try:
            return round(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2**20, 1)
        except Exception:  # noqa: BLE001
            return None

    def close(self) -> None:
        if self._nvml is not None:
            self._nvml.nvmlShutdown()


def _thirds(samples: list[dict[str, Any]], warmup_s: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The first and last third of the samples after the warm-up: the run's start and end."""
    steady = [s for s in samples if s["t_s"] > warmup_s and s["phase"] == "load"]
    n = len(steady) // 3
    return (steady[:n], steady[-n:]) if n else ([], [])


def _median(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def evaluate(samples: list[dict[str, Any]], final: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """The run's checks: name, whether it passed, and what was measured."""
    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    totals = final["totals"]
    check("no failed requests", totals["failed"] == 0,
          f"{totals['finished']} finished, {totals['failed']} failed ({final['errors'] or 'none'}), "
          f"{totals['cancelled']} cancelled by the client on purpose")
    check("invariant checks on", final["check_invariants"] is True,
          "every engine step checks the KV invariants" if final["check_invariants"] else
          f"this build does not check them (check_invariants = {final['check_invariants']})")
    alive = final["health_status"] == "ok" and final["probe_ok"] and not final["log_markers"]
    check("engine alive, no invariant violation", alive,
          f"/health {final['health_status']}; probe request {'ok' if final['probe_ok'] else 'failed'}; server log: "
          + (", ".join(f"{k} x{v}" for k, v in final["log_markers"].items()) or "no violation, no traceback"))
    kv = final["kv_after_drain"]
    leak_free = kv["used_pages"] == 0 and kv["shared_pages"] == 0 and \
        kv["free_pages"] + kv["cached_pages"] == kv["total_pages"]
    check("no KV pages held after the drain", leak_free,
          f"used {kv['used_pages']}, shared {kv['shared_pages']}, cached {kv['cached_pages']}, free {kv['free_pages']} "
          f"of {kv['total_pages']}; engine active {final['engine_after_drain'].get('active')}, "
          f"pending {final['engine_after_drain'].get('pending')}")

    early, late = _thirds(samples, args.warmup_min * 60)
    if not early:
        check("enough samples to compare start and end", False,
              f"{len(samples)} samples; the run is too short for its warm-up")
        return checks
    span = f"start = {early[0]['t_s']:.0f}–{early[-1]['t_s']:.0f} s, end = {late[0]['t_s']:.0f}–{late[-1]['t_s']:.0f} s"

    gpu_e, gpu_l = _median([s["gpu_mem_mib"] for s in early]), [s["gpu_mem_mib"] for s in late]
    if gpu_e is None:
        check("GPU memory flat", True, "not sampled (no NVML)")
    else:
        peak = max(v for v in gpu_l if v is not None)
        check("GPU memory flat", peak - gpu_e <= args.gpu_mem_tolerance_mib,
              f"start median {gpu_e:.0f} MiB, end peak {peak:.0f} MiB (allowed +{args.gpu_mem_tolerance_mib:.0f})")
    rss_e, rss_l = _median([s["rss_mib"] for s in early]), _median([s["rss_mib"] for s in late])
    if rss_e is None or rss_l is None:
        check("server memory flat", True, "not sampled (no /proc)")
    else:
        allowed = max(args.rss_tolerance_mib, rss_e * args.rss_tolerance_frac)
        check("server memory flat", rss_l - rss_e <= allowed,
              f"resident memory start median {rss_e:.0f} MiB, end median {rss_l:.0f} MiB (allowed +{allowed:.0f})")

    for key, label, tol in (("ttft_p50_ms", "TTFT P50", args.latency_tolerance),
                            ("ttft_p95_ms", "TTFT P95", args.latency_tolerance_p95),
                            ("itl_p50_ms", "ITL P50", args.latency_tolerance)):
        a, b = _median([s[key] for s in early]), _median([s[key] for s in late])
        if a is None or b is None:
            check(f"{label} stable", False, "no finished requests to compare")
            continue
        check(f"{label} stable", b <= a * (1 + tol),
              f"median of the sample windows: start {a:.1f} ms, end {b:.1f} ms ({b / a - 1:+.1%}; allowed +{tol:.0%})")
    a, b = _median([s["output_tokens_per_s"] for s in early]), _median([s["output_tokens_per_s"] for s in late])
    check("throughput stable", a is not None and b is not None and b >= a * (1 - args.throughput_tolerance),
          f"output tok/s start {a}, end {b}" + (f" ({b / a - 1:+.1%}; allowed -{args.throughput_tolerance:.0%})"
                                                 if a and b is not None else "") + f"; {span}")
    return checks


TIMELINE_FIELDS = ("t_s", "phase", "finished", "failed", "cancelled", "ttft_p50_ms", "ttft_p95_ms", "itl_p50_ms",
                   "output_tokens_per_s", "active", "pending", "used_pages", "cached_pages", "free_pages",
                   "shared_pages", "evicted_pages", "preemptions", "hit_rate", "gpu_mem_mib", "rss_mib")


def render_markdown(summary: dict[str, Any]) -> str:
    meta, final, checks, samples = summary["meta"], summary["final"], summary["checks"], summary["samples"]
    passed = all(c["passed"] for c in checks)
    lines = [
        "# Stress run", "",
        f"**Result: {'PASS' if passed else 'FAIL'}** ({sum(c['passed'] for c in checks)} of {len(checks)} checks)", "",
        f"- **When:** {meta['started_at']} to {meta['finished_at']} ({meta['git_commit']})",
        f"- **GPU:** {meta['gpu']} · **model:** {meta['model']} · **config:** `{meta['config']}` with prefix caching "
        f"on and {meta['pool_pages']} KV pages ({meta['pool_fraction']:.0%} of {meta['base_pages']})",
        f"- **Load:** {meta['concurrency']} clients in a closed loop for {meta['duration_min']:g} min (each sends its "
        f"next request when its last one ends); mix " + ", ".join(f"`{n}` {w:.0%}" for n, w in MIX) +
        f"; {meta['cancel_fraction']:.0%} of requests cancelled after 0.05–1.5 s; seed {meta['seed']}",
        f"- **Sampling:** every {meta['interval_s']:g} s; start and end compared over the first and last third of "
        f"the samples after a {meta['warmup_min']:g}-minute warm-up", "",
        "| Check | Result | Measured |", "| --- | --- | --- |",
    ]
    lines += [f"| {c['name']} | {'pass' if c['passed'] else '**FAIL**'} | {c['detail']} |" for c in checks]
    t = final["totals"]
    lines += ["", f"Totals: {t['finished']} requests finished, {t['output_tokens']} output tokens, "
                  f"{t['cancelled']} cancelled by the client, {t['failed']} failed; {final['evicted_pages']} pages "
                  f"evicted, {final['preemptions']} preemptions, prefix hit rate {final['hit_rate']:.1%}.", "",
              "## Per minute", "",
              "| Minute | finished | TTFT P50 | TTFT P95 | ITL P50 | output tok/s | used pages (max) | cached pages "
              "| evicted | preempted | GPU MiB | server RSS MiB |", "|" + " --- |" * 12]
    load = [s for s in samples if s["phase"] == "load"]
    prev = None
    for minute, group in itertools.groupby(load, key=lambda s: int((s["t_s"] - 1e-9) // 60)):
        g = list(group)
        last = g[-1]
        ev = last["evicted_pages"] - (prev["evicted_pages"] if prev else 0)
        pre = last["preemptions"] - (prev["preemptions"] if prev else 0)
        prev = last

        def med(k: str) -> str:
            return compare._fmt(_median([s[k] for s in g]))

        lines.append(f"| {minute + 1} | {sum(s['finished'] for s in g)} | {med('ttft_p50_ms')} | "
                     f"{med('ttft_p95_ms')} | "
                     f"{med('itl_p50_ms')} | {med('output_tokens_per_s')} | {max(s['used_pages'] for s in g)} | "
                     f"{med('cached_pages')} | {ev} | {pre} | {med('gpu_mem_mib')} | {med('rss_mib')} |")
    lines += ["", f"Latency cells are medians of the {meta['interval_s']:g} s windows in that minute (ms). "
                  "`timeline.csv` has every sample."]
    return "\n".join(lines) + "\n"


async def _client(base_url: str, source: Iterator[tuple[str, PlannedRequest]], stop: asyncio.Event,
                  done: list[tuple[float, loadgen.RequestResult]], t0: float, rng: random.Random,
                  cancel_fraction: float, counter: Iterator[int]) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=600) as client:
        while not stop.is_set():
            profile, planned = next(source)
            result = loadgen.RequestResult(index=next(counter), prompt_id=f"{profile}/{planned.prompt_id}",
                                           planned_at_s=round(time.monotonic() - t0, 6))
            task = asyncio.ensure_future(loadgen._send(client, planned, result, t0))
            if rng.random() < cancel_fraction:
                finished, _ = await asyncio.wait({task}, timeout=rng.uniform(0.05, 1.5))
                if not finished:
                    task.cancel()
                    result.error = "cancelled_by_client"
            try:
                await task
            except asyncio.CancelledError:
                result.error = "cancelled_by_client"
            done.append((time.monotonic() - t0, result))


async def _sample(base_url: str, pid: int, gpu: GpuMemory, done: list, t0: float, prev_t: float,
                  phase: str) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        health = (await client.get("/health")).json()
        state = (await client.get("/v1/runtime/state")).json()
    now = time.monotonic() - t0
    kv, engine = state["kv_cache"], health.get("engine", {})
    return {"t_s": round(now, 1), "phase": phase, **window_metrics(done, prev_t, now),
            "active": engine.get("active"), "pending": engine.get("pending"),
            **{k: kv[k] for k in ("used_pages", "cached_pages", "free_pages", "shared_pages", "evicted_pages")},
            "preemptions": state["preemptions"], "hit_rate": state["prefix_cache"]["hit_rate"],
            "gpu_mem_mib": gpu.used_mib(), "rss_mib": rss_mib(pid)}


async def run_load(base_url: str, pid: int, args: argparse.Namespace, model: str,
                   timeline: csv.DictWriter, timeline_file: Any) -> tuple[list, list]:
    source = request_source(MIX, args.seed, model)
    rng, counter = random.Random(f"neuroserve:stress:cancel:{args.seed}"), itertools.count()
    stop, done, samples = asyncio.Event(), [], []
    gpu = GpuMemory(args.gpu_index)
    t0 = time.monotonic()
    clients = [asyncio.ensure_future(_client(base_url, source, stop, done, t0, rng, args.cancel_fraction, counter))
               for _ in range(args.concurrency)]
    end = args.duration_min * 60
    prev = 0.0
    try:
        while True:
            phase = "load" if prev < end else "drain"
            await asyncio.sleep(min(args.interval_s, max(0.0, end - prev)) if phase == "load" else args.interval_s)
            sample = await _sample(base_url, pid, gpu, done, t0, prev, phase)
            prev = sample["t_s"]
            samples.append(sample)
            timeline.writerow(sample)
            timeline_file.flush()
            print(f"[{sample['t_s']:7.1f}s {phase}] finished={sample['finished']} failed={sample['failed']} "
                  f"ttft_p50={compare._fmt(sample['ttft_p50_ms'])} itl_p50={compare._fmt(sample['itl_p50_ms'])} "
                  f"tok/s={compare._fmt(sample['output_tokens_per_s'])} active={sample['active']} "
                  f"pending={sample['pending']} used={sample['used_pages']} cached={sample['cached_pages']} "
                  f"evicted={sample['evicted_pages']} preempt={sample['preemptions']} "
                  f"gpu={compare._fmt(sample['gpu_mem_mib'])}MiB rss={compare._fmt(sample['rss_mib'])}MiB", flush=True)
            if prev >= end and not stop.is_set():
                stop.set()  # clients finish the requests they have in flight, then stop
            if stop.is_set() and all(c.done() for c in clients):
                break
        for c in clients:
            c.result()  # re-raise a client's unexpected error
    finally:
        stop.set()
        gpu.close()
    return samples, done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neuroserve.bench.stress", description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/dev.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--duration-min", type=float, default=30.0)
    parser.add_argument("--warmup-min", type=float, default=5.0, help="excluded from the start/end comparison")
    parser.add_argument("--interval-s", type=float, default=10.0)
    parser.add_argument("--concurrency", type=int, default=24, help="requests kept in flight (3x max_batch)")
    parser.add_argument("--cancel-fraction", type=float, default=0.05)
    parser.add_argument("--pool-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-mem-tolerance-mib", type=float, default=64.0)
    parser.add_argument("--rss-tolerance-mib", type=float, default=64.0)
    parser.add_argument("--rss-tolerance-frac", type=float, default=0.05)
    parser.add_argument("--latency-tolerance", type=float, default=0.25, help="TTFT P50 and ITL P50")
    parser.add_argument("--latency-tolerance-p95", type=float, default=0.5)
    parser.add_argument("--throughput-tolerance", type=float, default=0.2)
    args = parser.parse_args(argv)
    if args.concurrency < 1 or args.duration_min <= 0 or args.interval_s <= 0:
        parser.error("concurrency, duration and interval must be positive")

    invariants = invariant_checks()
    base = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    base_cfg = ServerConfig.model_validate(base)
    cfg = prefix_ab.run_config(base, prefix_caching=True, pool_fraction=args.pool_fraction, burst=False,
                               requests=1, port=args.port)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": compare._git_commit(),
        "config": str(args.config), "model": base_cfg.model.name, "base_pages": base_cfg.engine.num_pages,
        "pool_pages": cfg["engine"]["num_pages"], "pool_fraction": args.pool_fraction,
        "concurrency": args.concurrency, "duration_min": args.duration_min, "warmup_min": args.warmup_min,
        "interval_s": args.interval_s, "cancel_fraction": args.cancel_fraction, "seed": args.seed,
        "mix": dict(MIX), "gpu": None, "thresholds": {k: getattr(args, k) for k in (
            "gpu_mem_tolerance_mib", "rss_tolerance_mib", "rss_tolerance_frac", "latency_tolerance",
            "latency_tolerance_p95", "throughput_tolerance")},
    }
    print(f"stress run: {args.concurrency} clients for {args.duration_min:g} min, {cfg['engine']['num_pages']} KV "
          f"pages, prefix caching on, invariant checks {'on' if invariants else 'OFF'}; a sample every "
          f"{args.interval_s:g} s", flush=True)
    server = prefix_ab.ServerProcess(cfg, args.out_dir, "stress")
    final: dict[str, Any] = {"check_invariants": invariants}
    with (args.out_dir / "timeline.csv").open("w", newline="", encoding="utf-8") as f:
        timeline = csv.DictWriter(f, fieldnames=TIMELINE_FIELDS)
        timeline.writeheader()
        try:
            meta["gpu"] = server.wait_ready(args.startup_timeout_s).get("gpu")
            samples, done = asyncio.run(run_load(server.url, server.pid, args, base_cfg.model.name, timeline, f))
            client = compare.Server(server.url)
            try:
                final["engine_after_drain"] = client.wait_idle(timeout_s=300)
                state = client.get("/v1/runtime/state")
                probe, _ = asyncio.run(loadgen.run_schedule(server.url, build_schedule(PROBE, 99, base_cfg.model.name,
                                                                                       num_requests=1)))
                final["probe_ok"] = all(r.ok for r in probe)
                final["health_status"] = client.get("/health").get("status")
            finally:
                client.close()
        finally:
            server.stop()
    log_text = server.log_path.read_text(encoding="utf-8", errors="replace") if server.log_path.exists() else ""
    final["log_markers"] = {m: log_text.count(m) for m in LOG_FAILURE_MARKERS if m in log_text}
    final["kv_after_drain"] = state["kv_cache"]
    final["evicted_pages"] = state["kv_cache"]["evicted_pages"]
    final["preemptions"] = state["preemptions"]
    final["hit_rate"] = state["prefix_cache"]["hit_rate"]
    results = [r for _, r in done]
    errors: dict[str, int] = {}
    for r in results:
        if not r.ok and r.error != "cancelled_by_client":
            errors[r.error or f"http_{r.status}"] = errors.get(r.error or f"http_{r.status}", 0) + 1
    final["errors"] = errors
    final["totals"] = {"finished": sum(r.ok for r in results), "failed": sum(errors.values()),
                       "cancelled": sum(r.error == "cancelled_by_client" for r in results),
                       "output_tokens": sum(r.completion_tokens or 0 for r in results if r.ok)}
    meta["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    checks = evaluate(samples, final, args)
    summary = {"meta": meta, "final": final, "checks": checks, "samples": samples}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    with (args.out_dir / "requests.jsonl").open("w", encoding="utf-8") as f:
        for t, r in done:
            f.write(json.dumps({"ended_at_s": round(t, 3), **asdict(r)}) + "\n")
    for c in checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']}: {c['detail']}")
    print(f"wrote {args.out_dir / 'summary.md'}, summary.json, timeline.csv and requests.jsonl")
    return 0 if all(c["passed"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
