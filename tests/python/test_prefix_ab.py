"""Prefix caching on vs off: run configs, the reuse bound, the counters, and full runs against
in-process servers (one per run, like the real comparison)."""

import itertools
import json

import httpx
import pytest
import yaml

from neuroserve.bench import prefix_ab
from neuroserve.bench.workload import PROFILES, build_schedule
from neuroserve.server.app import create_app
from neuroserve.server.config import ServerConfig
from neuroserve.server.engine_service import EngineService
from test_api import MODEL_NAME, live_server, make_app, tok  # noqa: F401  (tok is a fixture)

BASE = {"model": {"name": MODEL_NAME, "weights_dir": "/models/w", "tokenizer_dir": "/models/t"},
        "engine": {"num_pages": 2048, "max_queue": 256, "queue_timeout_ms": 30000},
        "server": {"host": "0.0.0.0", "port": 8000}}


def test_a_run_config_changes_only_the_cache_the_pool_and_for_a_burst_the_queue():
    before = json.dumps(BASE, sort_keys=True)
    paced = prefix_ab.run_config(BASE, prefix_caching=True, pool_fraction=0.25, burst=False, requests=120, port=8100)
    assert paced["engine"] == {"num_pages": 512, "max_queue": 256, "queue_timeout_ms": 30000, "prefix_caching": True}
    assert paced["server"] == {"host": "127.0.0.1", "port": 8100} and paced["model"] == BASE["model"]
    burst = prefix_ab.run_config(BASE, prefix_caching=False, pool_fraction=1.0, burst=True, requests=400, port=8100)
    assert burst["engine"] == {"num_pages": 2048, "max_queue": 400, "queue_timeout_ms": 0, "prefix_caching": False}
    assert json.dumps(BASE, sort_keys=True) == before  # the base config is not modified
    # Defaults come from the validated config when the base leaves them out.
    minimal = prefix_ab.run_config({"model": BASE["model"]}, prefix_caching=True, pool_fraction=0.25, burst=True,
                                   requests=10, port=9000)
    assert minimal["engine"]["num_pages"] == 512 and minimal["engine"]["max_queue"] == 256


def test_an_invalid_run_config_is_refused():
    with pytest.raises(ValueError):
        prefix_ab.run_config({**BASE, "engine": {"num_pages": 2048, "prefix_cachng": True}}, prefix_caching=True,
                             pool_fraction=1.0, burst=False, requests=1, port=8100)


def test_the_reuse_bound_counts_every_repeat_of_a_prefix():
    profile = PROFILES["shared_prefix"]
    schedule = build_schedule(profile, 1, MODEL_NAME, num_requests=200)
    distinct = len({p.prompt_id.split("+")[0] for p in schedule})
    assert distinct == profile.prefix_count
    # 384 tokens = 24 pages, reusable by every request but the first of each prefix.
    assert prefix_ab.reuse_upper_bound(profile, schedule, 16) == (200 - distinct) * 24


def _state(lookups, lookup_pages, hits, reused, evicted, preemptions, enabled=True):
    return {"kv_cache": {"evicted_pages": evicted, "total_pages": 512, "page_size_tokens": 16},
            "prefix_cache": {"enabled": enabled, "lookups": lookups, "lookup_pages": lookup_pages,
                             "hit_pages": hits, "reused_tokens": reused},
            "preemptions": preemptions}


def test_the_counters_of_a_run_are_differences():
    d = prefix_ab.kv_delta(_state(4, 10, 2, 32, 0, 1), _state(14, 110, 77, 1232, 30, 1))
    assert d == {"lookups": 10, "lookup_pages": 100, "hit_pages": 75, "reused_tokens": 1200, "hit_rate": 0.75,
                 "evicted_pages": 30, "preemptions": 0, "enabled": True, "total_pages": 512, "page_size_tokens": 16}


class FakeCachingServer:
    """An in-process server built from a run's config, like the real one: its KV pool and prefix
    switch come from the config, and with the cache on every request reuses its prefix's pages."""

    started: list = []
    tok = None
    ignore_prefix_switch = False

    def __init__(self, config, out_dir, name):
        cfg = ServerConfig.model_validate(config)
        enabled = cfg.engine.prefix_caching and not self.ignore_prefix_switch
        if self.ignore_prefix_switch:
            cfg = cfg.model_copy(update={"engine": cfg.engine.model_copy(update={"prefix_caching": False})})
        _, engine, _ = make_app(self.tok)
        counters = dict.fromkeys(("lookups", "lookup_pages", "hit_pages", "reused", "prefill"), 0)
        run, kv_stats, stats = engine.run, engine.kv_stats, engine.stats

        def caching_run(prompt_ids, max_new_tokens, **kw):
            counters["prefill"] += len(prompt_ids)
            if enabled:  # 384-token prefix + 64: 28 full pages looked up, 24 reused
                counters["lookups"] += 1
                counters["lookup_pages"] += len(prompt_ids) // 16
                counters["hit_pages"] += 24
                counters["reused"] += 384
            return run(prompt_ids, max_new_tokens, **kw)

        engine.run = caching_run
        engine.kv_stats = lambda: {**kv_stats(), "total_pages": cfg.engine.num_pages,
                                   "free_pages": cfg.engine.num_pages, "prefix_lookups": counters["lookups"],
                                   "prefix_lookup_pages": counters["lookup_pages"],
                                   "prefix_hit_pages": counters["hit_pages"],
                                   "prefix_reused_tokens": counters["reused"]}
        engine.stats = lambda: {**stats(), "prefill_tokens": counters["prefill"]}
        self._serving = live_server(create_app(cfg, EngineService(cfg, engine, self.tok, device_name="fake-gpu")))
        self.url = self._serving.__enter__()
        (out_dir / "configs").mkdir(parents=True, exist_ok=True)
        (out_dir / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(config))
        FakeCachingServer.started.append((name, config))

    def wait_ready(self, timeout_s):
        return httpx.get(f"{self.url}/health").json()

    def stop(self):
        self._serving.__exit__(None, None, None)


@pytest.fixture
def fake_servers(tok, tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(BASE))
    FakeCachingServer.started, FakeCachingServer.tok, FakeCachingServer.ignore_prefix_switch = [], tok, False
    monkeypatch.setattr(prefix_ab, "ServerProcess", FakeCachingServer)
    return base


def test_a_comparison_with_a_server_per_run(fake_servers, tmp_path):
    out = tmp_path / "out"
    code = prefix_ab.main(["--config", str(fake_servers), "--profiles", "shared_prefix", "--modes", "paced,burst",
                           "--requests", "3", "--rounds", "2", "--warmup-rounds", "1", "--cooldown-s", "0",
                           "--sample-interval-s", "0.05", "--out-dir", str(out)])
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())
    runs = summary["runs"]
    # A fresh server per run; the variant order alternates from round to round.
    assert len(FakeCachingServer.started) == len(runs) == 3 * 2 * 2
    assert [(r["round"], r["mode"], r["variant"]) for r in runs[:4]] == [
        (1, "paced", "cache-off"), (1, "paced", "cache-on"), (1, "burst", "cache-off"), (1, "burst", "cache-on")]
    assert [r["variant"] for r in runs if r["round"] == 2] == ["cache-on", "cache-off"] * 2
    assert [r["measured"] for r in runs] == [False] * 4 + [True] * 8
    for _, pair in itertools.groupby(runs, key=lambda r: (r["round"], r["mode"])):
        a, b = list(pair)
        assert a["schedule_sha256"] == b["schedule_sha256"] and a["seed"] == b["seed"] == a["round"]
    # Each server got the requested switch; a burst has no queue timeout.
    for name, config in FakeCachingServer.started:
        assert config["engine"]["prefix_caching"] == name.endswith("cache-on")
        assert config["engine"]["num_pages"] == 2048
        assert config["engine"]["queue_timeout_ms"] == (0 if "-burst-" in name else 30000)
    # The warm-up requests are not counted: 3 requests, 3 lookups.
    on = [r for r in runs if r["variant"] == "cache-on"]
    off = [r for r in runs if r["variant"] == "cache-off"]
    assert all(r["kv"]["lookups"] == 3 and r["values"]["hit_rate"] == round(24 / 28, 4) for r in on)
    assert all(r["values"]["prefilled_tokens"] == 3 * 448 - 3 * 384 for r in on)
    assert all(r["kv"]["lookups"] == 0 and r["values"]["prefilled_tokens"] == 3 * 448 for r in off)
    agg = summary["aggregate"]["shared_prefix/paced"]
    assert agg["cache-off"]["rounds"] == 2 and agg["cache-on"]["vs_baseline"]["prefilled_tokens"] == {
        "median_change": round(64 / 448 - 1, 4), "better_in": 2, "rounds": 2}
    report = (out / "summary.md").read_text()
    assert report.startswith("# Prefix caching on vs off")
    assert "## `shared_prefix`, burst:" in report and "| `cache-on` |" in report
    # 2 measured rounds x 3 requests x 28 pages looked up, 24 of them reused each time.
    assert "6 lookups; 144 of 168 full prompt pages reused (85.7%)" in report
    # The report can be recomputed from the saved runs alone.
    (out / "summary.md").unlink()
    stripped = {**summary, "runs": [{**r, "values": {}, "gpu": {}} for r in runs]}
    (out / "summary.json").write_text(json.dumps(stripped))
    assert prefix_ab.main(["--rerender", str(out)]) == 0
    again = json.loads((out / "summary.json").read_text())
    for a, b in zip(runs, again["runs"]):
        assert a["values"]["prefilled_tokens"] == b["values"]["prefilled_tokens"]
        assert a["values"]["ttft_p50_ms"] == b["values"]["ttft_p50_ms"]
    assert (out / "summary.md").exists()


def test_memory_pressure_runs_on_a_quarter_of_the_pool(fake_servers, tmp_path):
    code = prefix_ab.main(["--config", str(fake_servers), "--profiles", "memory_pressure", "--modes", "burst",
                           "--requests", "2", "--rounds", "1", "--warmup-rounds", "0", "--cooldown-s", "0",
                           "--out-dir", str(tmp_path / "out")])
    assert code == 0
    assert [c["engine"]["num_pages"] for _, c in FakeCachingServer.started] == [512, 512]
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert {r["kv"]["total_pages"] for r in summary["runs"]} == {512}


def test_a_server_that_ignores_the_switch_stops_the_comparison(fake_servers, tmp_path):
    FakeCachingServer.ignore_prefix_switch = True
    with pytest.raises(RuntimeError, match="not the requested config"):
        prefix_ab.main(["--config", str(fake_servers), "--profiles", "shared_prefix", "--modes", "burst",
                        "--requests", "2", "--rounds", "1", "--warmup-rounds", "0", "--cooldown-s", "0",
                        "--out-dir", str(tmp_path / "out")])


def test_only_prefix_profiles_are_accepted(tmp_path):
    with pytest.raises(SystemExit):
        prefix_ab.main(["--profiles", "mixed", "--out-dir", str(tmp_path)])
