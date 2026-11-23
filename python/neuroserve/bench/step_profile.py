"""Where an engine step's time goes: prefill, decode and the host (see docs/profiling.md).

    python -m neuroserve.bench.step_profile --config configs/dev.yaml --out-dir benchmarks/results/raw/profile-1

Starts a server from the config (or uses a running one with --url), then runs a few scenarios one
after another. Before each it resets the engine's step profile (DELETE /v1/runtime/profile); after
it, it reads every step back (GET /v1/runtime/profile?records=true). The engine times each step's
phases on the host: the backend's prefill and decode calls return their tokens to the host, so
their time includes the GPU work they launch. Writes summary.md, a CSV of steps and a Chrome /
Perfetto trace (trace-<scenario>.json, open it in ui.perfetto.dev) per scenario, and summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

from neuroserve.server.config import ServerConfig

from . import compare, loadgen, prefix_ab
from .gpu_monitor import GpuSampler
from .workload import PROFILES, Profile, build_schedule

# The scenarios: decode-bound, prefill-bound, and a realistic mix.
SCENARIOS: dict[str, tuple[Profile, str]] = {
    "decode": (Profile("decode", "32 requests at once, 64-token prompts, 256 output tokens each", rate_rps=1000.0,
                       num_requests=32, kind="tokens", prompt_tokens=((64, 1.0),), output_tokens=((256, 1.0),)),
               "the batch fills to max_batch and drains: decode time at every batch size"),
    "prefill": (Profile("prefill", "32 requests at once, 480-token prompts, 8 output tokens each", rate_rps=1000.0,
                        num_requests=32, kind="tokens", prompt_tokens=((480, 1.0),), output_tokens=((8, 1.0),)),
                "mostly prefill: its cost per token"),
    "mixed": (PROFILES["mixed"], "the mixed profile at its own rate: a realistic step mix"),
    # Day 52: short chats decoding while long prompts arrive; chunked prefill keeps their steps short.
    "interference": (Profile("interference", "48 requests at 3/s, 3 in 4 with 32-token prompts and 1 in 4 with "
                             "480, 128 output tokens each", rate_rps=3.0, num_requests=48, kind="tokens",
                             prompt_tokens=((32, 0.75), (480, 0.25)), output_tokens=((128, 1.0),)),
                     "decodes sharing steps with long prefills: the step-time tail and the time between tokens"),
}
WARMUP = prefix_ab.WARMUP
PHASES = ("housekeeping_ms", "schedule_ms", "preempt_ms", "prefill_ms", "decode_ms", "host_ms", "invariants_ms")
HOST_PHASES = ("housekeeping_ms", "schedule_ms", "preempt_ms", "host_ms")  # the loop's own work


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _pct(values: list[float], q: float) -> float | None:
    return loadgen.percentile(values, q) if values else None


def analyse(profile: dict[str, Any]) -> dict[str, Any]:
    """The split of one scenario's time, and the per-step costs it implies."""
    records, totals = profile.get("records", []), profile["totals"]
    busy = totals["total_ms"]
    out: dict[str, Any] = {
        "elapsed_ms": round(profile["elapsed_ms"], 1), "idle_ms": round(profile["idle_ms"], 1),
        "busy_ms": round(busy, 1), "steps": profile["steps"], "busy_steps": profile["busy_steps"],
        "dropped": profile["dropped"],
        "share": {p: round(totals[p] / busy, 4) if busy else 0.0 for p in PHASES},
        "gpu_share": round((totals["prefill_ms"] + totals["decode_ms"]) / busy, 4) if busy else 0.0,
        "host_share": round(sum(totals[p] for p in HOST_PHASES) / busy, 4) if busy else 0.0,
        "prefill_tokens": totals["prefill_tokens"], "decode_tokens": totals["decode_batch"],
    }
    # Decode cost by batch size, from steps that only decoded (no prefill in the same step).
    by_batch: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        if r["decode_batch"] > 0 and r["prefill_requests"] == 0:
            by_batch.setdefault(r["decode_batch"], []).append(r)
    out["decode_by_batch"] = [
        {"batch": b, "steps": len(rs), "decode_ms_p50": round(_median([r["decode_ms"] for r in rs]), 3),
         "decode_ms_p95": round(_pct([r["decode_ms"] for r in rs], 0.95), 3),
         "per_sequence_ms": round(_median([r["decode_ms"] for r in rs]) / b, 3),
         "host_ms_p50": round(_median([sum(r[p] for p in HOST_PHASES) for r in rs]), 3),
         "invariants_ms_p50": round(_median([r["invariants_ms"] for r in rs]), 3),
         "step_ms_p50": round(_median([r["total_ms"] for r in rs]), 3)}
        for b, rs in sorted(by_batch.items())]
    # Prefill cost per token, from steps that prefilled.
    pre = [r for r in records if r["prefill_requests"] > 0 and r["prefill_tokens"] > 0]
    out["prefill"] = ({
        "steps": len(pre), "tokens": sum(r["prefill_tokens"] for r in pre),
        "ms_per_token_p50": round(_median([r["prefill_ms"] / r["prefill_tokens"] for r in pre]), 4),
        "tokens_per_s": round(sum(r["prefill_tokens"] for r in pre) / (sum(r["prefill_ms"] for r in pre) / 1000), 0),
        "ms_per_request_p50": round(_median([r["prefill_ms"] / r["prefill_requests"] for r in pre]), 3),
    } if pre else None)
    steps = [r["total_ms"] for r in records if r["prefill_requests"] or r["decode_batch"]]
    out["step_ms_p50"] = round(_median(steps), 3) if steps else None
    out["step_ms_p99"] = round(_pct(steps, 0.99), 3) if steps else None
    return out


def chrome_trace(records: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    """The steps as a Chrome / Perfetto trace: one track, a slice per step and one per phase inside
    it. Phase order within a step is the loop's (prefill calls interleave with host work in reality;
    the host slice gathers all of it after decode)."""
    events: list[dict[str, Any]] = [{"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "neuroserve"}},
                                    {"name": "thread_name", "ph": "M", "pid": 1, "tid": 1,
                                     "args": {"name": f"engine thread ({scenario})"}}]
    for i, r in enumerate(records):
        start = r["start_ms"] * 1000.0  # microseconds
        events.append({"name": "step", "ph": "X", "pid": 1, "tid": 1, "ts": round(start, 3),
                       "dur": round(r["total_ms"] * 1000.0, 3),
                       "args": {"index": i, "decode_batch": r["decode_batch"],
                                "prefill_requests": r["prefill_requests"], "prefill_tokens": r["prefill_tokens"]}})
        t = start
        for phase in PHASES:
            dur = r[phase] * 1000.0
            if dur > 0:
                events.append({"name": phase[:-3], "ph": "X", "pid": 1, "tid": 1, "ts": round(t, 3),
                               "dur": round(dur, 3)})
            t += dur
    return {"traceEvents": events, "displayTimeUnit": "ms"}


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "–"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def _on_off(v: bool | None) -> str:
    return "unknown (a running server)" if v is None else "on" if v else "off"


def _chunk(v: int | None) -> str:
    return "unknown" if v is None else "off" if v == 0 else f"{v} tokens per prompt per step"


def render_markdown(summary: dict[str, Any]) -> str:
    meta = summary["meta"]
    lines = ["# Engine step profile", "",
             f"- **When:** {meta['started_at']} ({meta['git_commit']})",
             f"- **GPU:** {meta['gpu']} · **model:** {meta['model']} · **config:** `{meta['config']}`; prefix "
             f"caching {_on_off(meta['prefix_caching'])}; invariant checks "
             f"{_on_off(meta['check_invariants'])}; step budget {_fmt(meta.get('max_batch_tokens'))} tokens, "
             f"chunked prefill {_chunk(meta.get('prefill_chunk_tokens'))}",
             "- **Method:** the engine times every step's phases on its own thread (host clock). The backend's "
             "prefill and decode calls return tokens to the host, so their time includes the GPU work they "
             "launch and wait for; everything else in the step is host work. Idle = the engine thread waiting "
             "for requests.", "",
             "## Where the time went", "",
             "| Scenario | busy / elapsed | prefill | decode | host (loop) | invariant checks | steps | step P50 "
             "(ms) | step P99 (ms) |", "|" + " --- |" * 9]
    for name, s in summary["scenarios"].items():
        a = s["analysis"]
        sh = a["share"]
        lines.append(f"| `{name}` | {a['busy_ms'] / a['elapsed_ms']:.0%} of {a['elapsed_ms'] / 1000:.1f} s | "
                     f"{sh['prefill_ms']:.1%} | {sh['decode_ms']:.1%} | {a['host_share']:.1%} | "
                     f"{sh['invariants_ms']:.1%} | {a['busy_steps']} | {_fmt(a['step_ms_p50'])} | "
                     f"{_fmt(a['step_ms_p99'])} |")
    lines += ["", "Shares are of the busy time (the sum of every step). Host (loop) = housekeeping, scheduling, "
                  "preemption and the rest of the step outside the backend calls and the invariant checks.", "",
              "## Latency", "",
              "| Scenario | TTFT P50 (ms) | TTFT P99 (ms) | time between tokens P50 (ms) | P99 (ms) |",
              "|" + " --- |" * 5]
    for name, s in summary["scenarios"].items():
        lat = s.get("latency") or {}
        ttft, itl = lat.get("client_ttft_ms") or {}, lat.get("server_itl_ms") or {}
        lines.append(f"| `{name}` | {_fmt(ttft.get('p50'))} | {_fmt(ttft.get('p99'))} | {_fmt(itl.get('p50'))} | "
                     f"{_fmt(itl.get('p99'))} |")
    lines += ["", "TTFT is client-side; the time between tokens is each request's mean, from the server, its P50 "
                  "and P99 over requests.", ""]
    for name, s in summary["scenarios"].items():
        a, (profile, what) = s["analysis"], SCENARIOS[name]
        lines += [f"## `{name}`: {profile.description}", "", f"{what[0].upper() + what[1:]}.", ""]
        if a["decode_by_batch"]:
            lines += ["Decode-only steps by batch size (ms):", "",
                      "| batch | steps | decode P50 | decode P95 | per sequence | host P50 | invariants P50 "
                      "| step P50 |",
                      "|" + " --- |" * 8]
            lines += [f"| {d['batch']} | {d['steps']} | {d['decode_ms_p50']:.2f} | {d['decode_ms_p95']:.2f} | "
                      f"{d['per_sequence_ms']:.2f} | {d['host_ms_p50']:.3f} | {d['invariants_ms_p50']:.3f} | "
                      f"{d['step_ms_p50']:.2f} |" for d in a["decode_by_batch"]]
            lines.append("")
        if a["prefill"]:
            p = a["prefill"]
            lines += [f"Prefill: {p['tokens']} tokens in {p['steps']} steps; {p['ms_per_token_p50']:.4f} ms per token "
                      f"(P50 over steps), {p['tokens_per_s']:.0f} tokens/s overall; {p['ms_per_request_p50']:.2f} ms "
                      "per request (P50).", ""]
        g = s["gpu"]
        if g.get("available"):
            gs = g["summary"]
            lines += [f"GPU: SM clock median {_fmt(gs['sm_clock_mhz_median'], 0)} MHz under load, max "
                      f"{_fmt(gs['temp_c_max'], 0)} °C, mean power {_fmt(gs['power_w_mean'], 1)} W.", ""]
        lines += [f"Requests: {s['succeeded']} of {s['requests']} succeeded; trace: `trace-{name}.json`, steps: "
                  f"`steps-{name}.csv`" + (f"; {a['dropped']} older steps not kept" if a["dropped"] else "") + ".", ""]
    return "\n".join(lines).rstrip() + "\n"


def run_scenario(url: str, name: str, model: str, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    profile = SCENARIOS[name][0]
    schedule = build_schedule(profile, args.seed, model)
    client = compare.Server(url)
    try:
        client.wait_idle()
        httpx.delete(f"{url}/v1/runtime/profile", timeout=30).raise_for_status()
        with GpuSampler(args.gpu_index, args.sample_interval_s) as sampler:
            results, duration = asyncio.run(loadgen.run_schedule(url, schedule))
            client.wait_idle(timeout_s=600)
        prof = client.get("/v1/runtime/profile?records=true")
    finally:
        client.close()
    with (out_dir / f"steps-{name}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["start_ms", *PHASES, "total_ms", "prefill_requests", "prefill_tokens",
                                               "decode_batch"])
        writer.writeheader()
        for r in prof.get("records", []):
            writer.writerow({k: r[k] for k in writer.fieldnames})
    (out_dir / f"trace-{name}.json").write_text(json.dumps(chrome_trace(prof.get("records", []), name)),
                                                encoding="utf-8")
    gpu = sampler.result()
    latency = loadgen.summarize(results, duration)
    return {"requests": len(results), "succeeded": sum(r.ok for r in results), "duration_s": round(duration, 3),
            "analysis": analyse(prof), "gpu": {k: v for k, v in gpu.items() if k != "samples"},
            "latency": {k: latency[k] for k in ("client_ttft_ms", "server_itl_ms")}, "totals": prof["totals"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neuroserve.bench.step_profile", description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/dev.yaml"))
    parser.add_argument("--url", help="profile a running server instead of starting one")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS), help="comma-separated: " + ", ".join(SCENARIOS))
    parser.add_argument("--prefix-caching", action="store_true", help="start the server with prefix caching on")
    parser.add_argument("--prefill-chunk-tokens", type=int, default=None,
                        help="start the server with chunked prefill, at most this many tokens per prompt per step")
    parser.add_argument("--max-batch-tokens", type=int, default=None, help="start the server with this step budget")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--sample-interval-s", type=float, default=0.2)
    args = parser.parse_args(argv)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    if not scenarios or [s for s in scenarios if s not in SCENARIOS]:
        parser.error(f"scenarios: one or more of {', '.join(SCENARIOS)}")

    base = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    base_cfg = ServerConfig.model_validate(base)
    model = base_cfg.model.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    server = None
    chunk_tokens: int | None = None  # the started server's settings (unknown for a running server)
    step_budget: int | None = None
    url = args.url
    if url is None:
        cfg = prefix_ab.run_config(base, prefix_caching=args.prefix_caching, pool_fraction=1.0, burst=True,
                                   requests=max(p.num_requests for p, _ in SCENARIOS.values()), port=args.port)
        for key in ("prefill_chunk_tokens", "max_batch_tokens"):
            if getattr(args, key) is not None:
                cfg["engine"][key] = getattr(args, key)
        engine_cfg = ServerConfig.model_validate(cfg).engine  # raises if the combination is invalid
        chunk_tokens, step_budget = engine_cfg.prefill_chunk_tokens, engine_cfg.max_batch_tokens
        server = prefix_ab.ServerProcess(cfg, args.out_dir, "profile")
    meta: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": compare._git_commit(),
        "config": str(args.config), "model": model, "url": url, "seed": args.seed, "scenarios": scenarios,
        "prefix_caching": args.prefix_caching if server else None, "gpu": None,
        "prefill_chunk_tokens": chunk_tokens, "max_batch_tokens": step_budget,
    }
    from .stress import invariant_checks
    meta["check_invariants"] = invariant_checks()
    summary: dict[str, Any] = {"meta": meta, "scenarios": {}}
    try:
        if server is not None:
            meta["gpu"] = server.wait_ready(args.startup_timeout_s).get("gpu")
            url = server.url
        else:
            meta["gpu"] = httpx.get(f"{url}/health", timeout=10).json().get("gpu")
        asyncio.run(loadgen.run_schedule(url, build_schedule(WARMUP, 1, model)))  # CUDA and clock warm-up
        for name in scenarios:
            t = time.monotonic()
            print(f"scenario {name}: {SCENARIOS[name][0].description} ...", flush=True)
            s = run_scenario(url, name, model, args, args.out_dir)
            summary["scenarios"][name] = s
            a = s["analysis"]
            print(f"  {s['succeeded']}/{s['requests']} ok in {time.monotonic() - t:.0f} s; "
                  f"busy {a['busy_ms'] / 1000:.1f} "
                  f"of {a['elapsed_ms'] / 1000:.1f} s: prefill {a['share']['prefill_ms']:.1%}, decode "
                  f"{a['share']['decode_ms']:.1%}, host {a['host_share']:.1%}, invariants "
                  f"{a['share']['invariants_ms']:.1%}", flush=True)
    finally:
        if server is not None:
            server.stop()
    meta["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"wrote {args.out_dir / 'summary.md'}, summary.json, steps-*.csv and trace-*.json")
    return 0 if all(s["succeeded"] == s["requests"] for s in summary["scenarios"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
