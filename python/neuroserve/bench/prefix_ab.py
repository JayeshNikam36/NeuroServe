"""Prefix caching on vs off, on workloads that share prompt prefixes (see docs/benchmarking.md).

    python -m neuroserve.bench.prefix_ab --config configs/dev.yaml --out-dir benchmarks/results/raw/prefix-ab-1

Every run gets a server process of its own, started from the base config with prefix caching on
or off (and for memory_pressure the KV pool cut to 25% of its pages), so every run starts with an
empty cache and the two variants differ in that one setting only. Within a round both variants
run the same seeded schedule (paired), in an order that alternates from round to round; the first
round warms up the GPU and is discarded. Each profile runs paced (its own Poisson rate: latency
at a realistic load) and as a burst (every request at once: throughput at saturation). Every run
records the GPU's clocks, temperature and power (NVML) and the server's prefix-cache counters.
Results: one JSON file per run, summary.json and summary.md.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

from neuroserve.server.config import ServerConfig

from . import compare, loadgen
from .gpu_monitor import GpuSampler, summarize_samples
from .workload import PROFILES, PlannedRequest, Profile, build_schedule, schedule_digest

VARIANTS = ("cache-off", "cache-on")   # the first is the baseline of the paired comparison
MODES = ("paced", "burst")
BURST_RATE_RPS = 1000.0                # every request within a fraction of a second
POOL_FRACTION = {"memory_pressure": 0.25}  # share of the base config's KV pages a profile runs with
WARMUP = Profile("warmup", "per-server warm-up, unrelated prompts", rate_rps=50.0, num_requests=4,
                 kind="tokens", prompt_tokens=((64, 1.0),), output_tokens=((16, 1.0),))

# (key, label, lower is better). Per run: client latency, throughput and the cache's effect.
METRICS = (
    ("ttft_p50_ms", "TTFT P50", True), ("ttft_p95_ms", "TTFT P95", True),
    ("e2e_p50_ms", "e2e P50", True), ("e2e_p95_ms", "e2e P95", True),
    ("itl_p50_ms", "ITL P50", True), ("output_tokens_per_s", "output tok/s", False),
    ("prefilled_tokens", "prefilled tokens", True), ("hit_rate", "hit rate", False),
)
PAIRED = ("ttft_p50_ms", "ttft_p95_ms", "e2e_p50_ms", "e2e_p95_ms", "output_tokens_per_s", "prefilled_tokens")


def run_config(base: dict[str, Any], *, prefix_caching: bool, pool_fraction: float, burst: bool,
               requests: int, port: int) -> dict[str, Any]:
    """The server config of one run: the base config with prefix caching on or off and its KV pool
    scaled. A burst queues every request at once, so it gets room for all of them and no queue
    timeout (a request that waits is the point, not an error). Raises if the result is invalid."""
    engine_base = ServerConfig.model_validate(base).engine
    cfg = copy.deepcopy(base)
    engine = cfg.setdefault("engine", {})
    engine["prefix_caching"] = prefix_caching
    engine["num_pages"] = max(1, round(engine_base.num_pages * pool_fraction))
    if burst:
        engine["queue_timeout_ms"] = 0
        engine["max_queue"] = max(engine_base.max_queue, requests)
    cfg["server"] = {**(cfg.get("server") or {}), "host": "127.0.0.1", "port": port}
    ServerConfig.model_validate(cfg)
    return cfg


def reuse_upper_bound(profile: Profile, schedule: list[PlannedRequest], page_tokens: int) -> int:
    """Prompt pages a cache that never evicted could reuse: every request after the first with its
    prefix reuses the prefix's full pages. Requests that start while the first one is still being
    prefilled miss in practice; this bound ignores that."""
    seen: set[str] = set()
    pages = 0
    for planned in schedule:
        prefix = planned.prompt_id.split("+", 1)[0]
        if prefix in seen:
            pages += profile.prefix_tokens // page_tokens
        seen.add(prefix)
    return pages


def kv_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """The prefix-cache counters of one run, from GET /v1/runtime/state before and after it."""
    pb, pa = before["prefix_cache"], after["prefix_cache"]
    d = {k: pa[k] - pb[k] for k in ("lookups", "lookup_pages", "hit_pages", "reused_tokens")}
    d["hit_rate"] = round(d["hit_pages"] / d["lookup_pages"], 4) if d["lookup_pages"] else 0.0
    d["evicted_pages"] = after["kv_cache"]["evicted_pages"] - before["kv_cache"]["evicted_pages"]
    d["preemptions"] = after["preemptions"] - before["preemptions"]
    d["enabled"] = pa["enabled"]
    d["total_pages"] = after["kv_cache"]["total_pages"]
    d["page_size_tokens"] = after["kv_cache"]["page_size_tokens"]
    return d


def run_values(record: dict[str, Any], results: list[loadgen.RequestResult]) -> dict[str, Any]:
    """The compared numbers of one run. The engine's prefill counter counts every prompt token it
    admits, reused or not, so the tokens really prefilled are that minus the reused ones."""
    values = compare.run_metrics(results, record["duration_s"])
    values["prefilled_tokens"] = record["engine"]["prefill_tokens"] - record["kv"]["reused_tokens"]
    values["hit_rate"] = record["kv"]["hit_rate"]
    return values


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per profile, mode and variant: the median and range of each metric over the measured rounds,
    and the paired change against cache-off (same round, same requests)."""
    measured = [r for r in runs if r["measured"]]
    out: dict[str, Any] = {}
    for key in dict.fromkeys((r["profile"], r["mode"]) for r in measured):
        mine = [r for r in measured if (r["profile"], r["mode"]) == key]
        base = {r["round"]: r for r in mine if r["variant"] == VARIANTS[0]}
        group: dict[str, Any] = {}
        for variant in [v for v in VARIANTS if any(r["variant"] == v for r in mine)]:
            rows = [r for r in mine if r["variant"] == variant]
            entry: dict[str, Any] = {
                "rounds": len(rows), "failed": sum(r["values"]["failed"] for r in rows), "errors": {},
                "metrics": {}, "vs_baseline": {},
                "kv": {k: sum(r["kv"][k] for r in rows) for k in
                       ("lookups", "lookup_pages", "hit_pages", "reused_tokens", "evicted_pages", "preemptions")},
                "reuse_upper_bound_pages": sum(r["reuse_upper_bound_pages"] for r in rows),
                "total_pages": rows[0]["kv"]["total_pages"],
            }
            for r in rows:
                for code, n in r["values"].get("errors", {}).items():
                    entry["errors"][code] = entry["errors"].get(code, 0) + n
            for metric, _, lower_is_better in METRICS:
                vals = [r["values"][metric] for r in rows if r["values"][metric] is not None]
                entry["metrics"][metric] = ({"median": round(statistics.median(vals), 4), "min": min(vals),
                                             "max": max(vals)} if vals else None)
                if variant == VARIANTS[0] or metric not in PAIRED:
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
        out[f"{key[0]}/{key[1]}"] = group
    return out


def gpu_warnings(runs: list[dict[str, Any]]) -> list[str]:
    """compare.gpu_warnings, with profile/mode as the group and the variant as the label."""
    return compare.gpu_warnings([{**r, "profile": f"{r['profile']}/{r['mode']}", "policy": r["variant"]}
                                 for r in runs])


def _metric_cell(m: dict[str, Any] | None, key: str) -> str:
    if m is None:
        return "–"
    if key == "hit_rate":
        return f"{m['median']:.1%} ({m['min']:.1%}–{m['max']:.1%})"
    return f"{compare._fmt(m['median'])} ({compare._fmt(m['min'])}–{compare._fmt(m['max'])})"


def render_markdown(summary: dict[str, Any]) -> str:
    meta, agg = summary["meta"], summary["aggregate"]
    lines = [
        "# Prefix caching on vs off",
        "",
        f"- **When:** {meta['started_at']} ({meta['git_commit']})",
        f"- **GPU:** {meta['gpu']} · **model:** {meta['model']} · **base config:** `{meta['config']}` "
        f"({meta['base_pages']} KV pages)",
        f"- **Design:** {meta['measured_rounds']} measured rounds after {meta['warmup_rounds']} discarded "
        f"warm-up round(s); a fresh server per run (empty cache), warmed with {WARMUP.num_requests} unrelated "
        f"requests; both variants run the same seeded schedule within a round, in alternating order "
        f"(seeds {meta['seeds'][0]}–{meta['seeds'][-1]})",
        "- **Modes:** paced = the profile's own Poisson rate; burst = every request sent at once "
        f"({BURST_RATE_RPS:.0f}/s), no queue timeout: output tok/s is then the server's throughput",
        "- **Cells:** median over the measured rounds (min–max). Times in ms, measured by the client; ITL "
        "is each request's mean gap (server). Prefilled tokens = prompt tokens admitted minus those "
        "reused from the cache. Hit rate = reused pages ÷ full prompt pages looked up.",
        "",
    ]
    labels = {k: label for k, label, _ in METRICS}
    for key, variants in agg.items():
        profile_name, mode = key.split("/")
        p = PROFILES[profile_name]
        pool = next(iter(variants.values()))["total_pages"]
        lines += [f"## `{profile_name}`, {mode}: {p.description}", "",
                  f"KV pool: {pool} pages ({pool / meta['base_pages']:.0%} of the base config)."
                  + (f" Rate: {BURST_RATE_RPS:.0f}/s." if mode == "burst" else ""), ""]
        header = ["Variant"] + [labels[k] for k, _, _ in METRICS] + ["failed"]
        lines += ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
        for variant, entry in variants.items():
            cells = [f"`{variant}`"] + [_metric_cell(entry["metrics"][k], k) for k, _, _ in METRICS]
            errors = ", ".join(f"{code} {n}" for code, n in sorted(entry["errors"].items()))
            cells.append(f"{entry['failed']} ({errors})" if errors else str(entry["failed"]))
            lines.append("| " + " | ".join(cells) + " |")
        if VARIANTS[1] in variants:
            on = variants[VARIANTS[1]]
            lines += ["", f"`{VARIANTS[1]}` paired with `{VARIANTS[0]}` (same round, same requests): median "
                          "change, and in how many rounds it was better.", "",
                      "| " + " | ".join(labels[k] for k in PAIRED) + " |", "|" + " --- |" * len(PAIRED),
                      "| " + " | ".join(compare._change(on["vs_baseline"].get(k)) for k in PAIRED) + " |"]
            kv, bound = on["kv"], on["reuse_upper_bound_pages"]
            if kv["lookup_pages"]:
                text = (f"Cache, summed over the measured `{VARIANTS[1]}` runs: {kv['lookups']} lookups; "
                        f"{kv['hit_pages']} of {kv['lookup_pages']} full prompt pages reused "
                        f"({kv['hit_pages'] / kv['lookup_pages']:.1%})")
                if bound:
                    text += (f"; a cache that never evicted could have reused {bound} "
                             f"({kv['hit_pages'] / bound:.0%} of that reached)")
                text += (f"; {kv['reused_tokens']} prompt tokens not prefilled; {kv['evicted_pages']} pages "
                         f"evicted; {kv['preemptions']} preemptions.")
                lines += ["", text]
        lines.append("")
    lines += ["## GPU", ""]
    measured = [r for r in summary["runs"] if r["measured"]]
    if not any(r["gpu"]["available"] for r in measured):
        reason = next((r["gpu"]["reason"] for r in measured), None) or "no samples"
        lines += [f"Not sampled: {reason}.", ""]
    else:
        lines += ["Clock figures use the samples taken under load (GPU utilisation of 50% or more).", "",
                  "| Profile, mode | Variant | SM clock median (MHz) | SM clock P5 | max temp (°C) "
                  "| mean power (W) "
                  "| power-capped | thermal slowdown |", "|" + " --- |" * 8]
        for key, variants in agg.items():
            profile_name, mode = key.split("/")
            for variant in variants:
                mine = [r["gpu"]["summary"] for r in measured
                        if (r["profile"], r["mode"], r["variant"]) == (profile_name, mode, variant)]

                def med(k: str) -> float | None:
                    vals = [g[k] for g in mine if g[k] is not None]
                    return statistics.median(vals) if vals else None

                def worst(k: str, fn: Any) -> float | None:
                    vals = [g[k] for g in mine if g[k] is not None]
                    return fn(vals) if vals else None

                def share(v: float | None) -> str:
                    return f"{v:.0%}" if v is not None else "–"

                lines.append(f"| `{profile_name}`, {mode} | `{variant}` | "
                             f"{compare._fmt(med('sm_clock_mhz_median'))} | "
                             f"{compare._fmt(worst('sm_clock_mhz_p5', min))} | "
                             f"{compare._fmt(worst('temp_c_max', max))} | {compare._fmt(med('power_w_mean'))} | "
                             f"{share(med('power_cap_frac'))} | {share(worst('thermal_slowdown_frac', max))} |")
        lines.append("")
    warnings = summary["warnings"]
    lines += ["Warnings: " + ("none." if not warnings else ""), ""]
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
        runs.append({**record, "values": run_values(record, results), "gpu": gpu})
    meta = {**old["meta"], "rerendered_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    summary = {"meta": meta, "aggregate": aggregate(runs), "warnings": gpu_warnings(runs), "runs": runs}
    write_summary(out_dir, summary)
    return summary


class ServerProcess:
    """A server started from a config file, its output in a log file; stopped with SIGINT."""

    def __init__(self, config: dict[str, Any], out_dir: Path, name: str) -> None:
        port = config["server"]["port"]
        self.url = f"http://127.0.0.1:{port}"
        try:
            httpx.get(f"{self.url}/health", timeout=2)
        except httpx.HTTPError:
            pass
        else:
            raise RuntimeError(f"something already answers on port {port}: stop it, or pass --port")
        self.config_path = out_dir / "configs" / f"{name}.yaml"
        self.log_path = out_dir / "server-logs" / f"{name}.log"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        self._log = self.log_path.open("w", encoding="utf-8")
        self._proc = subprocess.Popen([sys.executable, "-m", "neuroserve.server", "--config", str(self.config_path)],
                                      stdout=self._log, stderr=subprocess.STDOUT)

    @property
    def pid(self) -> int:
        return self._proc.pid

    def wait_ready(self, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            if self._proc.poll() is not None:
                raise RuntimeError(f"the server exited with code {self._proc.returncode}; see {self.log_path}")
            try:
                resp = httpx.get(f"{self.url}/health", timeout=5)
                if resp.status_code == 200 and resp.json().get("status") == "ok":
                    return resp.json()
            except (httpx.HTTPError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"the server was not ready after {timeout_s:.0f} s; see {self.log_path}")
            time.sleep(0.5)

    def stop(self) -> None:
        if self._proc.poll() is None:
            self._proc.send_signal(signal.SIGINT)
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=10)
        self._log.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neuroserve.bench.prefix_ab", description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/dev.yaml"), help="the base server config")
    parser.add_argument("--profiles", default="shared_prefix,memory_pressure", help="comma-separated")
    parser.add_argument("--modes", default=",".join(MODES), help="paced, burst or both")
    parser.add_argument("--rounds", type=int, default=4, help="measured rounds (even: each order equally often)")
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
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if [p for p in profiles if p not in PROFILES or PROFILES[p].kind != "prefix"] or \
            [m for m in modes if m not in MODES] or args.rounds < 1 or args.warmup_rounds < 0:
        parser.error("profiles must be prefix profiles, modes paced and/or burst, and rounds >= 1")

    base = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    base_cfg = ServerConfig.model_validate(base)
    model = base_cfg.model.name
    runs_dir = args.out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    total_rounds = args.warmup_rounds + args.rounds
    total = total_rounds * len(profiles) * len(modes) * len(VARIANTS)
    meta: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": compare._git_commit(),
        "config": str(args.config), "base_pages": base_cfg.engine.num_pages, "model": model, "gpu": None,
        "profiles": profiles, "modes": modes, "variants": list(VARIANTS), "pool_fraction": POOL_FRACTION,
        "warmup_rounds": args.warmup_rounds, "measured_rounds": args.rounds,
        "seeds": [args.seed + r for r in range(total_rounds)], "requests_override": args.requests,
        "burst_rate_rps": BURST_RATE_RPS, "cooldown_s": args.cooldown_s,
    }
    print(f"{total} runs: {total_rounds} rounds x {len(profiles)} profiles x {len(modes)} modes x "
          f"{len(VARIANTS)} variants, a fresh server each" +
          (f" (round 1{'' if args.warmup_rounds == 1 else f'-{args.warmup_rounds}'} discarded)"
           if args.warmup_rounds else ""), flush=True)
    runs: list[dict[str, Any]] = []
    started = time.monotonic()
    for round_index in range(total_rounds):
        measured = round_index >= args.warmup_rounds
        order = list(VARIANTS) if round_index % 2 == 0 else list(reversed(VARIANTS))
        seed = args.seed + round_index
        for profile_name in profiles:
            profile = PROFILES[profile_name]
            for mode in modes:
                burst = mode == "burst"
                schedule = build_schedule(profile, seed, model, num_requests=args.requests,
                                          rate_rps=BURST_RATE_RPS if burst else None)
                for position, variant in enumerate(order):
                    name = f"r{round_index + 1:02d}-{profile_name}-{mode}-{variant}"
                    want_cache = variant == "cache-on"
                    cfg = run_config(base, prefix_caching=want_cache,
                                     pool_fraction=POOL_FRACTION.get(profile_name, 1.0), burst=burst,
                                     requests=len(schedule), port=args.port)
                    server = ServerProcess(cfg, args.out_dir, name)
                    try:
                        health = server.wait_ready(args.startup_timeout_s)
                        meta["gpu"] = meta["gpu"] or health.get("gpu")
                        asyncio.run(loadgen.run_schedule(server.url, build_schedule(WARMUP, 1, model)))
                        client = compare.Server(server.url)
                        try:
                            before = client.wait_idle()
                            state_before = client.get("/v1/runtime/state")
                            with GpuSampler(args.gpu_index, args.sample_interval_s) as sampler:
                                results, duration = asyncio.run(loadgen.run_schedule(server.url, schedule))
                            after = client.wait_idle(timeout_s=600)
                            state_after = client.get("/v1/runtime/state")
                        finally:
                            client.close()
                    finally:
                        server.stop()
                    kv = kv_delta(state_before, state_after)
                    if kv["enabled"] != want_cache or kv["total_pages"] != cfg["engine"]["num_pages"] or \
                            (not want_cache and kv["lookups"]):
                        raise RuntimeError(f"{name}: the server ran with {kv}, not the requested config")
                    gpu = sampler.result()
                    record = {
                        "name": name, "round": round_index + 1, "measured": measured, "profile": profile_name,
                        "mode": mode, "variant": variant, "position": position, "seed": seed,
                        "schedule_sha256": schedule_digest(schedule), "duration_s": round(duration, 3),
                        "engine": {k: after.get(k, 0) - before.get(k, 0) for k in compare.ENGINE_COUNTERS},
                        "kv": kv,
                        "reuse_upper_bound_pages": reuse_upper_bound(profile, schedule, kv["page_size_tokens"]),
                    }
                    record["values"] = run_values(record, results)
                    record["gpu"] = {k: v for k, v in gpu.items() if k != "samples"}
                    runs.append(record)
                    (runs_dir / f"{name}.json").write_text(json.dumps(
                        {**record, "gpu": gpu, "summary": loadgen.summarize(results, duration),
                         "requests": [asdict(r) for r in results]}, indent=2) + "\n", encoding="utf-8")
                    done = len(runs)
                    eta_min = (time.monotonic() - started) / done * (total - done) / 60
                    v, g = record["values"], gpu["summary"]
                    clock = f" sm={g['sm_clock_mhz_median']:.0f}MHz" if g["sm_clock_mhz_median"] else ""
                    temp = f" {g['temp_c_max']}C" if g["temp_c_max"] is not None else ""
                    print(f"[{done:3d}/{total}] round {round_index + 1}{'' if measured else ' (warm-up)'} "
                          f"{profile_name:<15} {mode:<5} {variant:<9} ok={v['succeeded']}/{v['requests']} "
                          f"ttft_p50={compare._fmt(v['ttft_p50_ms'])} ttft_p95={compare._fmt(v['ttft_p95_ms'])} "
                          f"tok/s={compare._fmt(v['output_tokens_per_s'])} hit={v['hit_rate']:.1%} "
                          f"evicted={kv['evicted_pages']}{clock}{temp}  (~{eta_min:.0f} min left)", flush=True)
                    time.sleep(args.cooldown_s)

    meta["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {"meta": meta, "aggregate": aggregate(runs), "warnings": gpu_warnings(runs), "runs": runs}
    write_summary(args.out_dir, summary)
    return 0 if all(r["values"]["failed"] == 0 for r in runs if r["measured"]) else 1


if __name__ == "__main__":
    sys.exit(main())
