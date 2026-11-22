"""The stress run: its request stream, windows, checks, and short runs against an in-process server."""

import argparse
import collections
import itertools
import json
import os

import httpx
import pytest
import yaml

from neuroserve.bench import loadgen, stress
from neuroserve.server.app import create_app
from neuroserve.server.config import ServerConfig
from neuroserve.server.engine_service import EngineService
from test_api import MODEL_NAME, live_server, make_app, tok  # noqa: F401  (tok is a fixture)

BASE = {"model": {"name": MODEL_NAME, "weights_dir": "/models/w", "tokenizer_dir": "/models/t"},
        "engine": {"num_pages": 2048}}


def test_the_request_stream_follows_the_mix_and_is_seeded():
    picks = list(itertools.islice(stress.request_source(stress.MIX, 1, MODEL_NAME), 3000))
    counts = collections.Counter(name for name, _ in picks)
    for name, weight in stress.MIX:
        assert counts[name] / 3000 == pytest.approx(weight, abs=0.04)
    again = list(itertools.islice(stress.request_source(stress.MIX, 1, MODEL_NAME), 50))
    assert [(n, p.body) for n, p in again] == [(n, p.body) for n, p in picks[:50]]
    other = list(itertools.islice(stress.request_source(stress.MIX, 2, MODEL_NAME), 50))
    assert [p.body for _, p in other] != [p.body for _, p in picks[:50]]
    # New schedules (new prefixes) keep coming: more than one seed's worth of shared_prefix prefixes.
    prefixes = {tuple(p.body["prompt"][:384]) for n, p in picks if n == "shared_prefix"}
    assert len(prefixes) > 4


def _result(ttft, itl, tokens, error=None):
    r = loadgen.RequestResult(index=0, prompt_id="p", planned_at_s=0.0, status=200, ttft_ms=ttft, e2e_ms=ttft,
                              completion_tokens=tokens, server={"itl_ms": itl}, error=error)
    return r


def test_a_window_counts_what_ended_in_it_and_keeps_cancels_apart():
    done = [(1.0, _result(10, 2, 100)), (2.0, _result(30, 4, 100)), (3.0, _result(5, 1, 50, "cancelled_by_client")),
            (4.0, _result(1, 1, 1, "queue_timeout")), (11.0, _result(99, 9, 100))]
    w = stress.window_metrics(done, 0.0, 10.0)
    # loadgen.percentile is nearest-rank: of two values, P50 is the lower and P95 the higher.
    assert w == {"finished": 2, "failed": 1, "cancelled": 1, "ttft_p50_ms": 10, "ttft_p95_ms": 30,
                 "itl_p50_ms": 2, "output_tokens_per_s": 20.0}
    assert stress.window_metrics(done, 10.0, 20.0)["finished"] == 1


def _args(**kw):
    defaults = dict(warmup_min=0.0, gpu_mem_tolerance_mib=64.0, rss_tolerance_mib=64.0, rss_tolerance_frac=0.05,
                    latency_tolerance=0.25, latency_tolerance_p95=0.5, throughput_tolerance=0.2)
    return argparse.Namespace(**{**defaults, **kw})


def _sample(t, ttft=100.0, itl=10.0, tok=500.0, gpu=4000.0, rss=900.0):
    return {"t_s": t, "phase": "load", "ttft_p50_ms": ttft, "ttft_p95_ms": ttft * 3, "itl_p50_ms": itl,
            "output_tokens_per_s": tok, "gpu_mem_mib": gpu, "rss_mib": rss}


def _final(**kw):
    final = {"totals": {"finished": 100, "failed": 0, "cancelled": 5, "output_tokens": 9000}, "errors": {},
             "check_invariants": True, "health_status": "ok", "probe_ok": True, "log_markers": {},
             "kv_after_drain": {"used_pages": 0, "shared_pages": 0, "cached_pages": 300, "free_pages": 212,
                                "total_pages": 512},
             "engine_after_drain": {"active": 0, "pending": 0}}
    final.update(kw)
    return final


def test_a_steady_run_passes_every_check():
    samples = [_sample(10.0 * i) for i in range(1, 31)]
    checks = stress.evaluate(samples, _final(), _args())
    assert all(c["passed"] for c in checks), checks
    assert [c["name"] for c in checks] == [
        "no failed requests", "invariant checks on", "engine alive, no invariant violation",
        "no KV pages held after the drain", "GPU memory flat", "server memory flat", "TTFT P50 stable",
        "TTFT P95 stable", "ITL P50 stable", "throughput stable"]


@pytest.mark.parametrize(("change", "failing"), [
    ({"rss": 1200.0}, "server memory flat"),
    ({"gpu": 4200.0}, "GPU memory flat"),
    ({"ttft": 200.0}, "TTFT P50 stable"),
    ({"itl": 15.0}, "ITL P50 stable"),
    ({"tok": 300.0}, "throughput stable"),
])
def test_drift_at_the_end_of_the_run_fails_its_check(change, failing):
    samples = [_sample(10.0 * i) for i in range(1, 21)] + [_sample(10.0 * i, **change) for i in range(21, 31)]
    failed = [c["name"] for c in stress.evaluate(samples, _final(), _args()) if not c["passed"]]
    assert failing in failed


def test_leaks_failures_and_violations_fail_their_checks():
    samples = [_sample(10.0 * i) for i in range(1, 31)]
    final = _final(totals={"finished": 99, "failed": 1, "cancelled": 0, "output_tokens": 1},
                   errors={"engine_stopped": 1}, log_markers={"violat": 1}, check_invariants=None,
                   kv_after_drain={"used_pages": 3, "shared_pages": 0, "cached_pages": 0, "free_pages": 509,
                                   "total_pages": 512})
    failed = {c["name"] for c in stress.evaluate(samples, final, _args()) if not c["passed"]}
    assert failed == {"no failed requests", "invariant checks on", "engine alive, no invariant violation",
                      "no KV pages held after the drain"}


class FakeServer:
    """An in-process server built from the run's config (its pool size), with a pid and a log."""

    started = []
    tok = None

    def __init__(self, config, out_dir, name):
        cfg = ServerConfig.model_validate(config)
        _, engine, _ = make_app(self.tok)
        kv_stats = engine.kv_stats
        engine.kv_stats = lambda: {**kv_stats(), "total_pages": cfg.engine.num_pages,
                                   "free_pages": cfg.engine.num_pages}
        self._serving = live_server(create_app(cfg, EngineService(cfg, engine, self.tok, device_name="fake-gpu")))
        self.url = self._serving.__enter__()
        self.pid = os.getpid()
        self.log_path = out_dir / "server-logs" / f"{name}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("started\n")
        FakeServer.started.append(config)

    def wait_ready(self, timeout_s):
        return httpx.get(f"{self.url}/health").json()

    def stop(self):
        self._serving.__exit__(None, None, None)


def test_a_short_stress_run_against_the_server(tok, tmp_path, monkeypatch):
    FakeServer.started, FakeServer.tok = [], tok
    monkeypatch.setattr(stress.prefix_ab, "ServerProcess", FakeServer)
    monkeypatch.setattr(stress, "invariant_checks", lambda: True)
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(BASE))
    out = tmp_path / "out"
    code = stress.main(["--config", str(base), "--out-dir", str(out), "--duration-min", "0.1", "--warmup-min", "0",
                        "--interval-s", "0.5", "--concurrency", "6", "--cancel-fraction", "0.3",
                        "--latency-tolerance", "10", "--latency-tolerance-p95", "10", "--throughput-tolerance",
                        "0.99", "--rss-tolerance-mib", "4096"])
    summary = json.loads((out / "summary.json").read_text())
    assert code == 0, summary["checks"]
    config = FakeServer.started[0]
    assert config["engine"]["prefix_caching"] is True and config["engine"]["num_pages"] == 512
    load = [s for s in summary["samples"] if s["phase"] == "load"]
    assert len(load) >= 10 and load[-1]["t_s"] == pytest.approx(6.0, abs=0.6)
    totals = summary["final"]["totals"]
    assert totals["finished"] > 6 and totals["failed"] == 0 and totals["cancelled"] > 0
    assert summary["final"]["probe_ok"] and summary["final"]["kv_after_drain"]["used_pages"] == 0
    lines = (out / "timeline.csv").read_text().splitlines()
    assert lines[0].startswith("t_s,phase,finished") and len(lines) == len(summary["samples"]) + 1
    assert sum(1 for _ in (out / "requests.jsonl").open()) == totals["finished"] + totals["cancelled"]
    report = (out / "summary.md").read_text()
    assert report.startswith("# Stress run\n\n**Result: PASS**") and "## Per minute" in report


def test_a_build_without_invariant_checks_fails_the_run(tok, tmp_path, monkeypatch):
    FakeServer.started, FakeServer.tok = [], tok
    monkeypatch.setattr(stress.prefix_ab, "ServerProcess", FakeServer)
    monkeypatch.setattr(stress, "invariant_checks", lambda: False)
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(BASE))
    code = stress.main(["--config", str(base), "--out-dir", str(tmp_path / "out"), "--duration-min", "0.02",
                        "--warmup-min", "0", "--interval-s", "0.3", "--concurrency", "2"])
    assert code == 1
    checks = json.loads((tmp_path / "out" / "summary.json").read_text())["checks"]
    assert [c["name"] for c in checks if not c["passed"]][0] == "invariant checks on"
