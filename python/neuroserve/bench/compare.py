"""Compares scheduling policies on the same workloads (see docs/benchmarking.md).

    python -m neuroserve.bench.compare --out-dir benchmarks/results/raw/compare-1

For every round, each profile is run once under each policy, all with the same seeded schedule
(paired). The policy order changes from round to round following a balanced design, so no policy
always runs first, last, or right after the same one (a GPU that warms up during a round would
otherwise favour whichever policy runs first). The first round warms up the GPU and the server
and is discarded. Each run records the GPU's clocks, temperature and power (NVML) and the
engine's counters. Results: one JSON file per run, summary.json and summary.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from . import loadgen
from .gpu_monitor import GpuSampler
from .workload import DATASET_SHA256, DATASET_VERSION, PROFILES, build_schedule, schedule_digest

DEFAULT_POLICIES = ("fcfs", "token-budget", "decode-first")
ENGINE_COUNTERS = ("completed", "rejected", "timed_out", "cancelled", "preemptions", "decode_steps",
                   "prefill_tokens", "decode_tokens")
CLOCK_TOLERANCE = 0.05  # a run whose median SM clock is this far from its profile's typical one is flagged

# (key, label, lower is better). The metrics compared between policies, per run.
METRICS = (
    ("ttft_p50_ms", "TTFT P50", True), ("ttft_p95_ms", "TTFT P95", True), ("ttft_p99_ms", "TTFT P99", True),
    ("itl_p50_ms", "ITL P50", True), ("itl_p95_ms", "ITL P95", True), ("max_itl_p95_ms", "max ITL P95", True),
    ("e2e_p50_ms", "e2e P50", True), ("e2e_p95_ms", "e2e P95", True), ("e2e_p99_ms", "e2e P99", True),
    ("queue_p95_ms", "queue P95", True), ("output_tokens_per_s", "output tok/s", False),
)


def balanced_orders(items: list[str]) -> list[list[str]]:
    """A Williams design: every item takes every position equally often, and every item follows
    every other item equally often (for an odd count this needs twice as many orders as items)."""
    n = len(items)
    first, low, high = [0], 1, n - 1
    while len(first) < n:  # 0, 1, n-1, 2, n-2, ...
        if len(first) % 2:
            first.append(low)
            low += 1
        else:
            first.append(high)
            high -= 1
    rows = [[(x + shift) % n for x in first] for shift in range(n)]
    if n % 2:
        rows += [row[::-1] for row in rows]
    return [[items[i] for i in row] for row in rows]


def run_metrics(results: list[loadgen.RequestResult], duration_s: float) -> dict[str, Any]:
    """The compared numbers of one run: client TTFT and e2e, server per-request ITL, queueing."""
    ok = [r for r in results if r.ok]

    def pct(values: list[float], q: float) -> float | None:
        value = loadgen.percentile(values, q)
        return round(value, 3) if value is not None else None

    ttft = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    e2e = [r.e2e_ms for r in ok if r.e2e_ms is not None]
    itl = [r.server["itl_ms"] for r in ok if "itl_ms" in r.server]           # mean gap, per request
    max_itl = [r.server["max_itl_ms"] for r in ok if "max_itl_ms" in r.server]  # its longest gap
    queue = [r.server["queue_ms"] for r in ok if "queue_ms" in r.server]
    tokens = sum(r.completion_tokens or 0 for r in ok)
    return {
        "requests": len(results), "succeeded": len(ok), "failed": len(results) - len(ok),
        "ttft_p50_ms": pct(ttft, 0.5), "ttft_p95_ms": pct(ttft, 0.95), "ttft_p99_ms": pct(ttft, 0.99),
        "itl_p50_ms": pct(itl, 0.5), "itl_p95_ms": pct(itl, 0.95), "max_itl_p95_ms": pct(max_itl, 0.95),
        "e2e_p50_ms": pct(e2e, 0.5), "e2e_p95_ms": pct(e2e, 0.95), "e2e_p99_ms": pct(e2e, 0.99),
        "queue_p95_ms": pct(queue, 0.95),
        "output_tokens_per_s": round(tokens / duration_s, 1) if duration_s > 0 else None,
    }


def aggregate(runs: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    """Per profile and policy: the median and range of each metric over the measured rounds, and
    the paired comparison with the baseline policy (same round, same schedule)."""
    measured = [r for r in runs if r["measured"]]
    out: dict[str, Any] = {}
    for profile in dict.fromkeys(r["profile"] for r in measured):
        out[profile] = {}
        by_round = {(r["round"], r["policy"]): r for r in measured if r["profile"] == profile}
        for policy in dict.fromkeys(r["policy"] for r in measured if r["profile"] == profile):
            mine = [r for r in measured if r["profile"] == profile and r["policy"] == policy]
            entry: dict[str, Any] = {"rounds": len(mine), "failed": sum(r["metrics"]["failed"] for r in mine),
                                     "preemptions": sum(r["engine"].get("preemptions", 0) for r in mine),
                                     "metrics": {}, "vs_baseline": {}}
            for key, _, lower_is_better in METRICS:
                values = [r["metrics"][key] for r in mine if r["metrics"][key] is not None]
                entry["metrics"][key] = ({"median": round(statistics.median(values), 3), "min": min(values),
                                          "max": max(values)} if values else None)
                if policy == baseline:
                    continue
                ratios, wins = [], 0
                for r in mine:
                    base = by_round.get((r["round"], baseline))
                    a, b = r["metrics"][key], base["metrics"][key] if base else None
                    if a is None or not b:
                        continue
                    ratios.append(a / b)
                    wins += (a < b) if lower_is_better else (a > b)
                entry["vs_baseline"][key] = ({"median_change": round(statistics.median(ratios) - 1.0, 4),
                                              "better_in": wins, "rounds": len(ratios)} if ratios else None)
            out[profile][policy] = entry
    return out


def gpu_warnings(runs: list[dict[str, Any]]) -> list[str]:
    """Runs whose GPU was throttled or clocked unlike the other runs of the same profile."""
    warnings = []
    measured = [r for r in runs if r["measured"] and r["gpu"]["summary"]["sm_clock_mhz_median"] is not None]
    for profile in dict.fromkeys(r["profile"] for r in measured):
        mine = [r for r in measured if r["profile"] == profile]
        typical = statistics.median(r["gpu"]["summary"]["sm_clock_mhz_median"] for r in mine)
        for r in mine:
            g = r["gpu"]["summary"]
            label = f"round {r['round']} {profile} {r['policy']}"
            if abs(g["sm_clock_mhz_median"] - typical) > CLOCK_TOLERANCE * typical:
                warnings.append(f"{label}: median SM clock {g['sm_clock_mhz_median']:.0f} MHz vs typical "
                                f"{typical:.0f} MHz")
            if g["thermal_slowdown_frac"]:
                warnings.append(f"{label}: thermal slowdown in {g['thermal_slowdown_frac']:.0%} of samples")
    return warnings


def _fmt(value: float | None) -> str:
    if value is None:
        return "–"
    if float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.0f}" if abs(value) >= 100 else f"{value:.1f}"


def render_markdown(summary: dict[str, Any]) -> str:
    meta, agg = summary["meta"], summary["aggregate"]
    baseline = meta["policies"][0]
    lines = [
        "# Scheduling policies compared",
        "",
        f"- **When:** {meta['started_at']} ({meta['git_commit']})",
        f"- **GPU:** {meta['gpu']} · **model:** {meta['model']} · **dataset:** {meta['dataset']['version']}",
        f"- **Design:** {meta['measured_rounds']} measured rounds after {meta['warmup_rounds']} discarded "
        f"warm-up round(s); policy order balanced across rounds; every policy runs the same seeded "
        f"schedule within a round (seeds {meta['seeds'][0]}–{meta['seeds'][-1]})",
        "- **Cells:** median over the measured rounds (min–max). Times in ms. TTFT and e2e are measured "
        "by the client; ITL is each request's mean time between tokens and max ITL its longest gap "
        "(server).",
        "",
    ]
    for profile, policies in agg.items():
        p = PROFILES[profile]
        lines += [f"## `{profile}`: {p.description}", ""]
        header = ["Policy"] + [label for _, label, _ in METRICS] + ["failed", "preempted"]
        lines += ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
        for policy, entry in policies.items():
            cells = [f"`{policy}`"]
            for key, _, _ in METRICS:
                m = entry["metrics"][key]
                cells.append(f"{_fmt(m['median'])} ({_fmt(m['min'])}–{_fmt(m['max'])})" if m else "–")
            cells += [str(entry["failed"]), str(entry["preemptions"])]
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", f"Paired with `{baseline}` (same round, same requests): median change, and in how many "
                      "rounds the policy was better.", ""]
        keys = [k for k in ("ttft_p50_ms", "ttft_p95_ms", "itl_p50_ms", "max_itl_p95_ms", "e2e_p50_ms",
                            "e2e_p95_ms", "output_tokens_per_s")]
        labels = {k: label for k, label, _ in METRICS}
        lines += ["| Policy | " + " | ".join(labels[k] for k in keys) + " |", "|" + " --- |" * (len(keys) + 1)]
        for policy, entry in policies.items():
            if policy == baseline:
                continue
            cells = [f"`{policy}`"]
            for k in keys:
                v = entry["vs_baseline"].get(k)
                cells.append(f"{v['median_change']:+.1%} ({v['better_in']}/{v['rounds']})" if v else "–")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## GPU", ""]
    gpu_rows = [r for r in summary["runs"] if r["measured"]]
    if not any(r["gpu"]["available"] for r in gpu_rows):
        reason = next((r["gpu"]["reason"] for r in gpu_rows), None) or "no samples"
        lines += [f"Not sampled: {reason}.", ""]
    else:
        lines += ["| Profile | Policy | SM clock median (MHz) | SM clock min | max temp (°C) | mean power (W) "
                  "| power-capped | thermal slowdown |", "|" + " --- |" * 8]
        for profile, policies in agg.items():
            for policy in policies:
                mine = [r["gpu"]["summary"] for r in gpu_rows if r["profile"] == profile and r["policy"] == policy]

                def med(key: str) -> float | None:
                    values = [g[key] for g in mine if g[key] is not None]
                    return statistics.median(values) if values else None

                def worst(key: str, fn: Any) -> float | None:
                    values = [g[key] for g in mine if g[key] is not None]
                    return fn(values) if values else None

                def share(value: float | None) -> str:
                    return f"{value:.0%}" if value is not None else "–"

                lines.append(f"| `{profile}` | `{policy}` | {_fmt(med('sm_clock_mhz_median'))} | "
                             f"{_fmt(worst('sm_clock_mhz_min', min))} | {_fmt(worst('temp_c_max', max))} | "
                             f"{_fmt(med('power_w_mean'))} | {share(med('power_cap_frac'))} | "
                             f"{share(worst('thermal_slowdown_frac', max))} |")
        lines.append("")
    warnings = summary["warnings"]
    lines += ["Warnings: " + ("none." if not warnings else ""), ""]
    lines += [f"- {w}" for w in warnings]
    return "\n".join(lines).rstrip() + "\n"


def check_policy(policy: str, decisions: list[dict[str, Any]]) -> None:
    """Every decision records the policy that made it: a run's must all be the requested one."""
    used = sorted({d["policy"] for d in decisions})
    if used != [policy]:
        raise RuntimeError(f"a run under {policy!r}: the decision log shows {used or 'no decisions'}")


def _git_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, check=True, timeout=10).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class Server:
    """The control calls the comparison makes to the server under test."""

    def __init__(self, base_url: str) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=30)

    def get(self, path: str) -> dict[str, Any]:
        resp = self._http.get(path)
        resp.raise_for_status()
        return resp.json()

    def set_policy(self, policy: str) -> None:
        """Requests the policy. The engine applies it at its next step, which for an idle engine is
        the next run's first request: whether the run really used it is checked afterwards, in the
        decision log (check_policy)."""
        resp = self._http.put("/v1/runtime/policy", json={"policy": policy})
        resp.raise_for_status()
        state = resp.json()
        if policy not in (state["policy"], state["pending"]):
            raise RuntimeError(f"asked for policy {policy!r}, the server reports {state}")

    def read_decisions(self, since: int) -> tuple[list[dict[str, Any]], int, int]:
        """Every scheduling decision after `since`: (records, next cursor, records missed)."""
        records, dropped = [], 0
        while True:
            page = self.get(f"/v1/runtime/decisions?since={since}&limit=1000")
            dropped += page["dropped"]
            if not page["data"]:
                return records, since, dropped
            records += page["data"]
            since = page["next_since"]

    def wait_idle(self, timeout_s: float = 120.0) -> dict[str, Any]:
        """Waits until nothing is queued or running; returns the engine's counters."""
        deadline = time.monotonic() + timeout_s
        while True:
            engine = self.get("/health")["engine"]
            if engine.get("pending", 0) == 0 and engine.get("active", 0) == 0:
                return engine
            if time.monotonic() > deadline:
                raise RuntimeError(f"the server is still busy after {timeout_s:.0f} s: {engine}")
            time.sleep(0.2)

    def close(self) -> None:
        self._http.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neuroserve.bench.compare", description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES),
                        help="comma-separated; the first is the baseline of the paired comparison")
    parser.add_argument("--profiles", default="short,mixed", help="comma-separated workload profiles")
    parser.add_argument("--rounds", type=int, default=6, help="measured rounds (6 balances 3 policies fully)")
    parser.add_argument("--warmup-rounds", type=int, default=1, help="rounds run first and discarded")
    parser.add_argument("--seed", type=int, default=1, help="round r (from 1) uses seed + r - 1")
    parser.add_argument("--requests", type=int, help="override each profile's number of requests")
    parser.add_argument("--rate", type=float, help="override each profile's arrival rate (requests/s)")
    parser.add_argument("--cooldown-s", type=float, default=5.0, help="idle pause after every run")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--sample-interval-s", type=float, default=0.2)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    unknown = [p for p in profiles if p not in PROFILES]
    if unknown or len(set(policies)) != len(policies) or len(policies) < 2 or args.rounds < 1 \
            or args.warmup_rounds < 0:
        parser.error(f"need 2+ distinct policies, known profiles (unknown: {unknown}) and rounds >= 1")
    orders = balanced_orders(policies)
    if args.rounds % len(orders):
        print(f"note: {args.rounds} rounds do not balance the order fully (a multiple of {len(orders)} does)")

    server = Server(args.url)
    health = server.get("/health")
    model = server.get("/v1/models")["data"][0]["id"]
    available = server.get("/v1/runtime/policy")["available"]
    if missing := [p for p in policies if p not in available]:
        parser.error(f"the server has no policy {missing} (available: {available})")
    original_policy = health["engine"]["policy"]
    runs_dir = args.out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    total_rounds = args.warmup_rounds + args.rounds
    total = total_rounds * len(profiles) * len(policies)
    meta = {"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": _git_commit(),
            "url": args.url, "model": model, "gpu": health.get("gpu"), "policies": policies, "profiles": profiles,
            "warmup_rounds": args.warmup_rounds, "measured_rounds": args.rounds,
            "seeds": [args.seed + r for r in range(total_rounds)], "orders": orders,
            "requests_override": args.requests, "rate_override": args.rate, "cooldown_s": args.cooldown_s,
            "dataset": {"version": DATASET_VERSION, "sha256": DATASET_SHA256},
            "server_at_start": {"engine": health.get("engine"), "kv_cache": health.get("kv_cache")}}
    print(f"{total} runs: {total_rounds} rounds x {len(profiles)} profiles x {len(policies)} policies "
          f"(round 1{'' if args.warmup_rounds == 1 else f'-{args.warmup_rounds}'} discarded)"
          if args.warmup_rounds else f"{total} runs")
    runs: list[dict[str, Any]] = []
    cursor = 0
    started = time.monotonic()
    try:
        for round_index in range(total_rounds):
            measured = round_index >= args.warmup_rounds
            order = orders[(round_index - args.warmup_rounds) % len(orders)] if measured else orders[0]
            seed = args.seed + round_index
            for profile_name in profiles:
                schedule = build_schedule(PROFILES[profile_name], seed, model, num_requests=args.requests,
                                          rate_rps=args.rate)
                for position, policy in enumerate(order):
                    server.set_policy(policy)
                    before = server.wait_idle()
                    _, cursor, _ = server.read_decisions(cursor)  # skip to the newest decision
                    with GpuSampler(args.gpu_index, args.sample_interval_s) as sampler:
                        results, duration = asyncio.run(loadgen.run_schedule(args.url, schedule))
                    after = server.wait_idle()
                    gpu = sampler.result()
                    decisions, cursor, dropped = server.read_decisions(cursor)
                    check_policy(policy, decisions)
                    record = {
                        "round": round_index + 1, "measured": measured, "profile": profile_name, "policy": policy,
                        "position": position, "seed": seed, "schedule_sha256": schedule_digest(schedule),
                        "duration_s": round(duration, 3), "metrics": run_metrics(results, duration),
                        "engine": {k: after.get(k, 0) - before.get(k, 0) for k in ENGINE_COUNTERS},
                        "decisions": {"count": len(decisions), "dropped": dropped},
                        "gpu": {k: v for k, v in gpu.items() if k != "samples"},
                    }
                    runs.append(record)
                    name = f"r{round_index + 1:02d}-{profile_name}-{policy}.json"
                    (runs_dir / name).write_text(json.dumps(
                        {**record, "gpu": gpu, "summary": loadgen.summarize(results, duration),
                         "requests": [asdict(r) for r in results]}, indent=2) + "\n", encoding="utf-8")
                    done = len(runs)
                    eta_min = (time.monotonic() - started) / done * (total - done) / 60
                    m, g = record["metrics"], gpu["summary"]
                    clock = f" sm={g['sm_clock_mhz_median']:.0f}MHz" if g["sm_clock_mhz_median"] else ""
                    temp = f" {g['temp_c_max']}C" if g["temp_c_max"] is not None else ""
                    print(f"[{done:3d}/{total}] round {round_index + 1}{'' if measured else ' (warm-up)'} "
                          f"{profile_name:<6} {policy:<13} ok={m['succeeded']}/{m['requests']} "
                          f"ttft_p95={_fmt(m['ttft_p95_ms'])} itl_p50={_fmt(m['itl_p50_ms'])} "
                          f"e2e_p95={_fmt(m['e2e_p95_ms'])}{clock}{temp}  (~{eta_min:.0f} min left)", flush=True)
                    time.sleep(args.cooldown_s)
    finally:
        try:
            server.set_policy(original_policy)  # leave the server as it was
        finally:
            server.close()

    meta["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {"meta": meta, "aggregate": aggregate(runs, policies[0]), "warnings": gpu_warnings(runs), "runs": runs}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"wrote {args.out_dir / 'summary.md'} and summary.json")
    return 0 if all(r["metrics"]["failed"] == 0 for r in runs if r["measured"]) else 1


if __name__ == "__main__":
    sys.exit(main())
