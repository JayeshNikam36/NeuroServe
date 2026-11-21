"""Load generator: reproducible schedules (no server needed) and a run against the API server."""

import asyncio
import json
import statistics
from collections import Counter

import pytest

from neuroserve.bench import loadgen
from neuroserve.bench.workload import (DATASET_SHA256, PROFILES, TOKEN_ID_RANGE, Profile, build_schedule,
                                       load_dataset, schedule_digest)
from test_api import MODEL_NAME, live_server, make_app, tok  # noqa: F401  (tok is a fixture)


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_the_same_seed_gives_the_same_schedule(name):
    a = build_schedule(PROFILES[name], seed=7, model=MODEL_NAME)
    b = build_schedule(PROFILES[name], seed=7, model=MODEL_NAME)
    assert a == b and schedule_digest(a) == schedule_digest(b)
    assert schedule_digest(build_schedule(PROFILES[name], seed=8, model=MODEL_NAME)) != schedule_digest(a)


def test_a_shorter_run_is_a_prefix_of_a_longer_one():
    profile = PROFILES["mixed"]
    assert build_schedule(profile, 3, MODEL_NAME, num_requests=50) == \
        build_schedule(profile, 3, MODEL_NAME, num_requests=100)[:50]


def test_arrivals_are_a_poisson_process():
    profile = Profile("poisson", "test", rate_rps=10.0, num_requests=5000, kind="chat")
    times = [p.at_s for p in build_schedule(profile, 1, MODEL_NAME)]
    gaps = [b - a for a, b in zip([0.0] + times, times)]
    assert all(g > 0 for g in gaps)                                     # strictly increasing
    assert statistics.mean(gaps) == pytest.approx(0.1, rel=0.05)         # mean gap = 1 / rate
    # Exponential gaps have a standard deviation equal to their mean (coefficient of variation 1);
    # evenly spaced arrivals would have 0.
    assert statistics.stdev(gaps) / statistics.mean(gaps) == pytest.approx(1.0, rel=0.1)


def test_mixed_prompts_have_exact_lengths_and_forced_outputs():
    schedule = build_schedule(PROFILES["mixed"], 1, MODEL_NAME, num_requests=2000)
    lengths = [len(p.body["prompt"]) for p in schedule]
    assert set(lengths) == {32, 128, 480}
    assert lengths.count(32) / len(lengths) == pytest.approx(0.5, abs=0.05)   # the profile's weights
    for p in schedule:
        assert p.endpoint == "/v1/completions" and p.prompt_id == f"tokens-{len(p.body['prompt'])}"
        assert all(TOKEN_ID_RANGE[0] <= t < TOKEN_ID_RANGE[1] for t in p.body["prompt"])
        assert p.body["max_tokens"] == p.max_tokens in (32, 128, 384)
        assert p.body["neuroserve"] == {"ignore_eos": True} and p.body["stream"] is True


@pytest.mark.parametrize("name", ["shared_prefix", "memory_pressure"])
def test_prefix_prompts_share_a_prefix_and_have_suffixes_of_their_own(name):
    profile = PROFILES[name]
    schedule = build_schedule(profile, 1, MODEL_NAME, num_requests=2000)
    prefixes = {}
    for p in schedule:
        prefix, suffix = p.prompt_id.split("+")
        ids = p.body["prompt"]
        assert p.endpoint == "/v1/completions" and len(ids) == profile.prefix_tokens + int(suffix)
        assert len(ids) <= 512 and all(TOKEN_ID_RANGE[0] <= t < TOKEN_ID_RANGE[1] for t in ids)
        assert prefixes.setdefault(prefix, ids[:profile.prefix_tokens]) == ids[:profile.prefix_tokens]
        assert p.body["neuroserve"] == {"ignore_eos": True} and p.body["stream"] is True
    assert len(prefixes) == profile.prefix_count
    assert len({tuple(v) for v in prefixes.values()}) == profile.prefix_count       # all distinct
    assert len({tuple(p.body["prompt"]) for p in schedule}) > 1990                  # suffixes differ
    assert profile.prefix_tokens % 16 == 0                                          # whole pages
    counts = Counter(p.prompt_id.split("+")[0] for p in schedule)
    if profile.prefix_zipf:  # weights 1 / (k + 1): prefix 0 about twice as often as prefix 1
        assert counts["prefix-0"] / counts["prefix-1"] == pytest.approx(2.0, rel=0.2)
    else:
        assert max(counts.values()) / min(counts.values()) < 1.3
    # Another seed, other prefixes.
    other = build_schedule(profile, 2, MODEL_NAME, num_requests=1)[0].body["prompt"][:profile.prefix_tokens]
    assert tuple(other) not in {tuple(v) for v in prefixes.values()}


def test_the_prompt_dataset_is_versioned():
    entries = load_dataset()  # raises if the content no longer matches the pinned digest
    assert len(entries) == 32 and len({e["id"] for e in entries}) == 32
    assert len(DATASET_SHA256) == 64


def test_dry_run_prints_the_same_fingerprint_for_the_same_seed(capsys):
    loadgen.main(["--profile", "mixed", "--seed", "5", "--dry-run"])
    first = capsys.readouterr().out
    loadgen.main(["--profile", "mixed", "--seed", "5", "--dry-run"])
    assert capsys.readouterr().out == first and "schedule=" in first


def test_a_run_against_the_server_sends_everything_on_time(tok, tmp_path):
    app, engine, _ = make_app(tok)
    with live_server(app) as base_url:
        out = tmp_path / "run.json"
        code = loadgen.main(["--url", base_url, "--profile", "short", "--seed", "1", "--requests", "12",
                             "--rate", "40", "--out", str(out)])
    report = json.loads(out.read_text())
    assert code == 0
    summary = report["summary"]
    assert summary["succeeded"] == 12 and summary["errors"] == {}
    assert summary["max_send_lag_ms"] < 500            # sent at their planned wall-clock times
    assert report["meta"]["model"] == MODEL_NAME and report["meta"]["server"]["model"] == MODEL_NAME
    assert report["meta"]["schedule_sha256"] == schedule_digest(
        build_schedule(PROFILES["short"], 1, MODEL_NAME, num_requests=12, rate_rps=40))
    for r in report["requests"]:
        assert r["status"] == 200 and r["completion_tokens"] == 64      # ignore_eos forced the length
        assert 0 < r["ttft_ms"] <= r["e2e_ms"] and r["server"]["itl_ms"] == 0.05
    assert all(max_new == 64 for _, max_new in engine.calls)


def test_failed_requests_are_counted_not_hidden(tok):
    import neuroserve

    app, _, _ = make_app(tok, error=neuroserve.QueueFull("the queue is full"))
    schedule = build_schedule(PROFILES["short"], 1, MODEL_NAME, num_requests=3, rate_rps=50)
    with live_server(app) as base_url:
        results, duration = asyncio.run(loadgen.run_schedule(base_url, schedule))
    summary = loadgen.summarize(results, duration)
    assert summary["succeeded"] == 0 and summary["errors"] == {"queue_full": 3}
    assert summary["client_ttft_ms"]["count"] == 0
