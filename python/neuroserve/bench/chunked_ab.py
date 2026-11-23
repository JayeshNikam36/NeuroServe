"""Chunked prefill against unchunked, with long prompts arriving (Day 54; docs/prefill-attention.md).

    python -m neuroserve.bench.chunked_ab --config configs/dev.yaml --out-dir benchmarks/results/raw/chunked-ab-1

Every run gets a server process of its own, started from the base config with one variant's
engine settings, so the variants differ in those settings only:

- `unchunked`: each prompt prefilled in one step (prefill_chunk_tokens 0), mixed steps on;
- `chunked`: at most --chunk-tokens of a prompt per step and a --chunk-budget step budget, so no
  step holds a whole long prefill, mixed steps on;
- `two-pass` (opt-in): unchunked with mixed steps off, the Day 51 engine.

Within a round every variant runs the same seeded schedule (paired), in an order that follows a
balanced design across rounds (compare.balanced_orders): every variant takes every position
equally often. The first round warms up the GPU and is discarded. Runs are paced (each profile's
own Poisson rate): latency at a realistic load is the question. The time between tokens is the
engine's, per token (its histogram, read from /health after the run; a fresh server per run makes
it this run's, plus the warm-up's few tokens), and per request its longest gap: its P99 is what a
streaming user feels as a hitch. Not the client's gaps between text chunks: a token that ends
inside a character, or an output the detokenizer holds back, streams no text of its own, and a
random-token prompt's output can stream a 128-token answer in a few chunks. TTFT is split by
prompt length, since chunking delays long prompts to spare short ones.
Results: one JSON file per run, summary.json and summary.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from neuroserve.server.config import ServerConfig

from . import compare, loadgen, prefix_ab
from .gpu_monitor import GpuSampler, summarize_samples
from .workload import PROFILES, build_schedule, schedule_digest

VARIANTS = ("unchunked", "chunked", "two-pass")  # the first is the baseline of the paired comparison
DEFAULT_VARIANTS = ("unchunked", "chunked")
DEFAULT_PROFILES = ("long_prompt", "mixed")
LONG_PROMPT_TOKENS = 256  # TTFT is reported for prompts of at least this many tokens and for the rest

# (key, label, lower is better)
METRICS = (
    ("ttft_p50_ms", "TTFT P50", True), ("ttft_p99_ms", "TTFT P99", True),
    ("ttft_long_p50_ms", "TTFT long P50", True), ("ttft_long_p99_ms", "TTFT long P99", True),
    ("ttft_short_p50_ms", "TTFT short P50", True), ("ttft_short_p99_ms", "TTFT short P99", True),
    ("itl_p50_ms", "ITL P50", True), ("itl_p99_ms", "ITL P99", True), ("itl_max_ms", "ITL max", True),
    ("text_gap_p99_ms", "text gap P99", True),
    ("stall_p99_ms", "worst gap P99", True), ("e2e_p50_ms", "e2e P50", True),
    ("output_tokens_per_s", "output tok/s", False),
)
PAIRED = ("ttft_p50_ms", "ttft_p99_ms", "ttft_long_p50_ms", "ttft_short_p50_ms", "itl_p99_ms", "stall_p99_ms",
          "e2e_p50_ms", "output_tokens_per_s")


def variant_engine(variant: str, chunk_tokens: int, chunk_budget: int) -> dict[str, Any]:
    """The engine settings a variant changes in the base config."""
    if variant == "unchunked":
        return {"prefill_chunk_tokens": 0, "mixed_steps": True}
    if variant == "chunked":
        return {"prefill_chunk_tokens": chunk_tokens, "max_batch_tokens": chunk_budget, "mixed_steps": True}
    if variant == "two-pass":
        return {"prefill_chunk_tokens": 0, "mixed_steps": False}
    raise ValueError(f"unknown variant {variant!r}")


def run_values(results: list[loadgen.RequestResult], duration_s: float,
               latency: dict[str, Any] | None = None) -> dict[str, Any]:
    """The compared numbers of one run; `latency` is the server's /health latency block (the engine's
    per-token histogram), None for a run that did not record it."""
    ok = [r for r in results if r.ok]

    def pct(values: list[float], q: float) -> float | None:
        value = loadgen.percentile(values, q)
        return round(value, 3) if value is not None else None

    def ttft(rows: list[loadgen.RequestResult]) -> list[float]:
        return [r.ttft_ms for r in rows if r.ttft_ms is not None]

    long_ = [r for r in ok if (r.prompt_tokens or 0) >= LONG_PROMPT_TOKENS]
    short = [r for r in ok if (r.prompt_tokens or 0) < LONG_PROMPT_TOKENS]
    text_gaps = [g for r in ok for g in r.token_gaps_ms]  # client-side, between text chunks
    stalls = [r.server["max_itl_ms"] for r in ok if "max_itl_ms" in r.server]  # each request's longest gap
    itl = (latency or {}).get("itl_ms") or {}

    def engine(key: str) -> float | None:
        return round(itl[key], 3) if itl.get("count") else None

    e2e = [r.e2e_ms for r in ok if r.e2e_ms is not None]
    tokens = sum(r.completion_tokens or 0 for r in ok)
    errors: dict[str, int] = {}
    for r in results:
        if not r.ok:
            code = r.error or f"http_{r.status}"
            errors[code] = errors.get(code, 0) + 1
    return {
        "requests": len(results), "succeeded": len(ok), "failed": len(results) - len(ok), "errors": errors,
        "long_requests": len(long_), "engine_tokens": itl.get("count"),
        "ttft_p50_ms": pct(ttft(ok), 0.5), "ttft_p99_ms": pct(ttft(ok), 0.99),
        "ttft_long_p50_ms": pct(ttft(long_), 0.5), "ttft_long_p99_ms": pct(ttft(long_), 0.99),
        "ttft_short_p50_ms": pct(ttft(short), 0.5), "ttft_short_p99_ms": pct(ttft(short), 0.99),
        "itl_p50_ms": engine("p50"), "itl_p99_ms": engine("p99"), "itl_max_ms": engine("max"),
        "text_gap_p99_ms": pct(text_gaps, 0.99),
        "stall_p99_ms": pct(stalls, 0.99), "e2e_p50_ms": pct(e2e, 0.5),
        "output_tokens_per_s": round(tokens / duration_s, 1) if duration_s > 0 else None,
    }


def aggregate(runs: list[dict[str, Any]], variants: list[str]) -> dict[str, Any]:
    """Per profile and variant: the median and range of each metric over the measured rounds, and
    the paired change against the first variant (same round, same requests)."""
    measured = [r for r in runs if r["measured"]]
    baseline = variants[0]
    out: dict[str, Any] = {}
    for profile in dict.fromkeys(r["profile"] for r in measured):
        mine_all = [r for r in measured if r["profile"] == profile]
        base = {r["round"]: r for r in mine_all if r["variant"] == baseline}
        group: dict[str, Any] = {}
        for variant in [v for v in variants if any(r["variant"] == v for r in mine_all)]:
            rows = [r for r in mine_all if r["variant"] == variant]
            entry: dict[str, Any] = {"rounds": len(rows), "failed": sum(r["values"]["failed"] for r in rows),
                                     "errors": {}, "metrics": {}, "vs_baseline": {},
                                     "mixed_steps": sum(r["engine"]["mixed_steps"] for r in rows),
                                     "max_step_tokens": max(r["engine"]["max_step_tokens"] for r in rows)}
            for r in rows:
                for code, n in r["values"].get("errors", {}).items():
                    entry["errors"][code] = entry["errors"].get(code, 0) + n
            for metric, _, lower_is_better in METRICS:
                vals = [r["values"][metric] for r in rows if r["values"][metric] is not None]
                entry["metrics"][metric] = ({"median": round(statistics.median(vals), 3), "min": min(vals),
                                             "max": max(vals)} if vals else None)
                if variant == baseline or metric not in PAIRED:
                    continue
                ratios, wins = [], 0
                for r in rows:
                    b = base.get(r["round"])
                    a, bv = r["values"][metric], b["values"][metric] if b else None
                    if a is None or not bv:
                        continue
                    ratios.append(a / bv)
                    wins += (a < bv) if lower_is_better else (a > bv)
                entry["vs_baseline"][metric] = ({"median_change": round(statistics.median(ratios) - 1.0, 4),
                                                 "better_in": wins, "rounds": len(ratios)} if ratios else None)
            group[variant] = entry
        out[profile] = group
    return out


def gpu_warnings(runs: list[dict[str, Any]]) -> list[str]:
    """compare.gpu_warnings, with the variant as the label."""
    return compare.gpu_warnings([{**r, "policy": r["variant"]} for r in runs])


def _cell(m: dict[str, Any] | None) -> str:
    if m is None:
        return "–"
    return f"{compare._fmt(m['median'])} ({compare._fmt(m['min'])}–{compare._fmt(m['max'])})"


def render_markdown(summary: dict[str, Any]) -> str:
    meta, agg = summary["meta"], summary["aggregate"]
    variants = meta["variants"]
    lines = [
        "# Chunked prefill against unchunked, long prompts arriving",
        "",
        f"- **When:** {meta['started_at']} ({meta['git_commit']})",
        f"- **GPU:** {meta['gpu']} · **model:** {meta['model']} · **base config:** `{meta['config']}`",
        "- **Variants:** " + "; ".join(f"`{v}` = {json.dumps(meta['variant_engine'][v])}" for v in variants),
        f"- **Design:** {meta['measured_rounds']} measured rounds after {meta['warmup_rounds']} discarded "
        f"warm-up round(s); a fresh server per run, warmed with {prefix_ab.WARMUP.num_requests} unrelated "
        f"requests; every variant runs the same seeded schedule within a round, the order balanced across "
        f"rounds (seeds {meta['seeds'][0]}–{meta['seeds'][-1]}); paced at each profile's own rate",
        "- **Cells:** median over the measured rounds (min–max). Times in ms, measured by the client. "
        f"TTFT long / short: prompts of at least / under {LONG_PROMPT_TOKENS} tokens. ITL: the engine's "
        "time between consecutive tokens of a request, over every token of the run; ITL max: the longest; "
        "worst gap P99: the P99 over requests of each request's longest gap (engine). Text gap P99: the "
        "client's gaps between streamed text chunks, for reference (an output can stream many tokens as one "
        "chunk).",
        "",
    ]
    labels = {k: label for k, label, _ in METRICS}
    for profile, entries in agg.items():
        lines += [f"## `{profile}`: {PROFILES[profile].description}", ""]
        header = ["Variant"] + [labels[k] for k, _, _ in METRICS] + ["failed", "mixed steps"]
        lines += ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
        for variant, entry in entries.items():
            errors = ", ".join(f"{code} {n}" for code, n in sorted(entry["errors"].items()))
            cells = [f"`{variant}`"] + [_cell(entry["metrics"][k]) for k, _, _ in METRICS]
            cells += [f"{entry['failed']} ({errors})" if errors else str(entry["failed"]), str(entry["mixed_steps"])]
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", f"Paired with `{variants[0]}` (same round, same requests): median change, and in how many "
                      "rounds the variant was better.", "",
                  "| Variant | " + " | ".join(labels[k] for k in PAIRED) + " |", "|" + " --- |" * (len(PAIRED) + 1)]
        for variant, entry in entries.items():
            if variant == variants[0]:
                continue
            lines.append(f"| `{variant}` | " + " | ".join(compare._change(entry["vs_baseline"].get(k))
                                                         for k in PAIRED) + " |")
        lines.append("")
    warnings = summary["warnings"]
    lines += ["## GPU", "", "Warnings: " + ("none." if not warnings else ""), ""]
    lines += [f"- {w}" for w in warnings]
    return "\n".join(lines).rstrip() + "\n"


def write_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"wrote {out_dir / 'summary.md'} and summary.json")


def rerender(out_dir: Path) -> dict[str, Any]:
    """Recomputes summary.json and summary.md from the saved runs, without repeating one."""
    old = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    runs = []
    for record in old["runs"]:
        saved = json.loads((out_dir / "runs" / f"{record['name']}.json").read_text(encoding="utf-8"))
        results = [loadgen.RequestResult(**r) for r in saved["requests"]]
        gpu = {k: v for k, v in saved["gpu"].items() if k != "samples"}
        gpu["summary"] = summarize_samples(saved["gpu"]["samples"])
        runs.append({**record, "values": run_values(results, record["duration_s"], record.get("engine_latency")),
                     "gpu": gpu})
    meta = {**old["meta"], "rerendered_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    summary = {"meta": meta, "aggregate": aggregate(runs, meta["variants"]), "warnings": gpu_warnings(runs),
               "runs": runs}
    write_summary(out_dir, summary)
    return summary


def check_run(name: str, variant: str, engine: dict[str, Any], settings: dict[str, Any]) -> None:
    """Raises if the server did not run with the variant's settings, as far as its counters show."""
    if variant == "two-pass" and engine["mixed_steps"]:
        raise RuntimeError(f"{name}: {engine['mixed_steps']} mixed steps with mixed steps off")
    if variant == "chunked" and engine["max_step_tokens"] > settings["max_batch_tokens"]:
        raise RuntimeError(f"{name}: a step of {engine['max_step_tokens']} tokens over the "
                           f"{settings['max_batch_tokens']}-token budget")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neuroserve.bench.chunked_ab", description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/dev.yaml"), help="the base server config")
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES), help="comma-separated")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
                        help="comma-separated, the first is the baseline: " + ", ".join(VARIANTS))
    parser.add_argument("--chunk-tokens", type=int, default=128, help="the chunked variant's prefill_chunk_tokens")
    parser.add_argument("--chunk-budget", type=int, default=256, help="the chunked variant's max_batch_tokens")
    parser.add_argument("--rounds", type=int, default=4,
                        help="measured rounds (a multiple of the number of orders balances them)")
    parser.add_argument("--warmup-rounds", type=int, default=1, help="rounds run first and discarded")
    parser.add_argument("--seed", type=int, default=1, help="round r (from 1) uses seed + r - 1")
    parser.add_argument("--requests", type=int, help="override each profile's number of requests")
    parser.add_argument("--port", type=int, default=8100, help="port of the servers this starts")
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--cooldown-s", type=float, default=5.0, help="pause after every run")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--sample-interval-s", type=float, default=0.2)
    parser.add_argument("--out-dir", type=Path, help="where to write the runs and the report")
    parser.add_argument("--rerender", type=Path, metavar="DIR",
                        help="recompute the report in DIR from its saved runs, without running anything")
    args = parser.parse_args(argv)
    if args.rerender is not None:
        rerender(args.rerender)
        return 0
    if args.out_dir is None:
        parser.error("--out-dir is required (or --rerender DIR)")
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if [p for p in profiles if p not in PROFILES] or [v for v in variants if v not in VARIANTS] or \
            len(set(variants)) != len(variants) or len(variants) < 2 or args.rounds < 1 or args.warmup_rounds < 0:
        parser.error(f"profiles from {', '.join(PROFILES)}; two or more distinct variants from "
                     f"{', '.join(VARIANTS)}; rounds >= 1")

    base = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    base_cfg = ServerConfig.model_validate(base)
    model = base_cfg.model.name
    settings = {v: variant_engine(v, args.chunk_tokens, args.chunk_budget) for v in variants}
    orders = compare.balanced_orders(variants)
    runs_dir = args.out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    total_rounds = args.warmup_rounds + args.rounds
    total = total_rounds * len(profiles) * len(variants)
    meta: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": compare._git_commit(),
        "config": str(args.config), "model": model, "gpu": None, "profiles": profiles, "variants": variants,
        "variant_engine": settings, "orders": orders, "warmup_rounds": args.warmup_rounds,
        "measured_rounds": args.rounds, "seeds": [args.seed + r for r in range(total_rounds)],
        "requests_override": args.requests, "cooldown_s": args.cooldown_s,
    }
    if args.rounds % len(orders):
        print(f"note: {args.rounds} measured rounds do not balance {len(orders)} orders", flush=True)
    print(f"{total} runs: {total_rounds} rounds x {len(profiles)} profiles x {len(variants)} variants, a fresh "
          "server each" + (f" (first {args.warmup_rounds} round(s) discarded)" if args.warmup_rounds else ""),
          flush=True)
    runs: list[dict[str, Any]] = []
    started = time.monotonic()
    for round_index in range(total_rounds):
        measured = round_index >= args.warmup_rounds
        order = orders[(round_index - args.warmup_rounds) % len(orders)] if measured else orders[0]
        seed = args.seed + round_index
        for profile_name in profiles:
            schedule = build_schedule(PROFILES[profile_name], seed, model, num_requests=args.requests)
            for position, variant in enumerate(order):
                name = f"r{round_index + 1:02d}-{profile_name}-{variant}"
                cfg = prefix_ab.run_config(base, prefix_caching=False, pool_fraction=1.0, burst=False,
                                           requests=len(schedule), port=args.port)
                cfg["engine"].update(settings[variant])
                ServerConfig.model_validate(cfg)  # raises if the variant's settings are invalid here
                server = prefix_ab.ServerProcess(cfg, args.out_dir, name)
                try:
                    health = server.wait_ready(args.startup_timeout_s)
                    meta["gpu"] = meta["gpu"] or health.get("gpu")
                    asyncio.run(loadgen.run_schedule(server.url, build_schedule(prefix_ab.WARMUP, 1, model)))
                    client = compare.Server(server.url)
                    try:
                        before = client.wait_idle()
                        with GpuSampler(args.gpu_index, args.sample_interval_s) as sampler:
                            results, duration = asyncio.run(loadgen.run_schedule(server.url, schedule))
                        after = client.wait_idle(timeout_s=600)
                        latency = client.get("/health")["latency"]  # the engine's histograms, this server's
                    finally:
                        client.close()
                finally:
                    server.stop()
                engine = {k: after.get(k, 0) - before.get(k, 0) for k in (*compare.ENGINE_COUNTERS, "mixed_steps")}
                engine["max_step_tokens"] = after.get("max_step_tokens", 0)
                check_run(name, variant, engine, settings[variant])
                gpu = sampler.result()
                record = {
                    "name": name, "round": round_index + 1, "measured": measured, "profile": profile_name,
                    "variant": variant, "position": position, "seed": seed,
                    "schedule_sha256": schedule_digest(schedule), "duration_s": round(duration, 3),
                    "engine": engine, "engine_latency": latency, "values": run_values(results, duration, latency),
                    "gpu": {k: v for k, v in gpu.items() if k != "samples"},
                }
                runs.append(record)
                (runs_dir / f"{name}.json").write_text(json.dumps(
                    {**record, "gpu": gpu, "requests": [asdict(r) for r in results]}, indent=2) + "\n",
                    encoding="utf-8")
                done = len(runs)
                eta_min = (time.monotonic() - started) / done * (total - done) / 60
                v, g = record["values"], gpu["summary"]
                clock = f" sm={g['sm_clock_mhz_median']:.0f}MHz" if g["sm_clock_mhz_median"] else ""
                print(f"[{done:3d}/{total}] round {round_index + 1}{'' if measured else ' (warm-up)'} "
                      f"{profile_name:<12} {variant:<10} ok={v['succeeded']}/{v['requests']} "
                      f"ttft_p50={compare._fmt(v['ttft_p50_ms'])} ttft_long_p50={compare._fmt(v['ttft_long_p50_ms'])} "
                      f"itl_p99={compare._fmt(v['itl_p99_ms'])} itl_max={compare._fmt(v['itl_max_ms'])} "
                      f"mixed={engine['mixed_steps']}{clock}  (~{eta_min:.0f} min left)", flush=True)
                time.sleep(args.cooldown_s)

    meta["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {"meta": meta, "aggregate": aggregate(runs, variants), "warnings": gpu_warnings(runs), "runs": runs}
    write_summary(args.out_dir, summary)
    return 0 if all(r["values"]["failed"] == 0 for r in runs if r["measured"]) else 1


if __name__ == "__main__":
    sys.exit(main())
