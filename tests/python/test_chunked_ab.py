"""The chunked-prefill A/B's numbers and report (Day 54), from made-up runs: no server, no GPU."""

import pytest

from neuroserve.bench import chunked_ab
from neuroserve.bench.loadgen import RequestResult
from neuroserve.server.config import ServerConfig


def _result(i, prompt_tokens, ttft, gaps, ok=True, max_itl=None):
    server = {"max_itl_ms": max_itl if max_itl is not None else max(gaps)} if gaps else {}
    return RequestResult(index=i, prompt_id=f"p{i}", planned_at_s=0.0, status=200 if ok else 503,
                         error=None if ok else "overloaded", ttft_ms=ttft, e2e_ms=ttft + sum(gaps),
                         prompt_tokens=prompt_tokens, completion_tokens=len(gaps) + 1, token_gaps_ms=list(gaps),
                         server=server)


def test_run_values_split_ttft_by_prompt_length_and_take_the_gaps_from_the_engine():
    # Request 1 streamed its last tokens as one text chunk (a 150 ms client gap); the engine's gaps are short.
    results = [_result(0, 32, 50.0, [10.0, 12.0]), _result(1, 480, 400.0, [10.0, 150.0], max_itl=30.0),
               _result(2, 32, 60.0, [11.0]), _result(3, 480, 0.0, [], ok=False)]
    latency = {"itl_ms": {"count": 4, "mean": 15.0, "p50": 11.0, "p95": 28.0, "p99": 29.5, "max": 30.0}}
    v = chunked_ab.run_values(results, duration_s=2.0, latency=latency)
    assert (v["requests"], v["succeeded"], v["failed"], v["errors"]) == (4, 3, 1, {"overloaded": 1})
    assert v["long_requests"] == 1 and v["ttft_long_p50_ms"] == 400.0
    assert v["ttft_short_p50_ms"] == 50.0  # nearest rank, as loadgen.percentile
    assert v["engine_tokens"] == 4 and (v["itl_p50_ms"], v["itl_p99_ms"], v["itl_max_ms"]) == (11.0, 29.5, 30.0)
    assert v["stall_p99_ms"] == 30.0  # the requests' longest gaps (engine): 12, 30, 11
    assert v["text_gap_p99_ms"] == 150.0  # the client's, between text chunks
    assert v["output_tokens_per_s"] == 4.0  # (3 + 3 + 2) tokens in 2 s
    assert chunked_ab.run_values(results, 2.0)["itl_p99_ms"] is None  # a run that did not record it


def _run(round_, variant, ttft, itl_p99, measured=True):
    values = {key: None for key, _, _ in chunked_ab.METRICS}
    values.update(failed=0, errors={}, ttft_p50_ms=ttft, itl_p99_ms=itl_p99, output_tokens_per_s=100.0)
    return {"name": f"r{round_}-long_prompt-{variant}", "round": round_, "measured": measured,
            "profile": "long_prompt", "variant": variant, "values": values,
            "engine": {"mixed_steps": 5, "max_step_tokens": 256}, "gpu": {"available": False, "summary": {}}}


def test_aggregate_pairs_each_round_with_the_baseline():
    runs = [_run(1, "unchunked", 900.0, 150.0, measured=False), _run(1, "chunked", 1.0, 1.0, measured=False)]
    for r, (base_ttft, chunk_ttft) in enumerate([(100.0, 120.0), (110.0, 121.0)], start=2):
        runs += [_run(r, "unchunked", base_ttft, 150.0), _run(r, "chunked", chunk_ttft, 60.0)]
    agg = chunked_ab.aggregate(runs, ["unchunked", "chunked"])["long_prompt"]
    assert agg["unchunked"]["rounds"] == 2 and agg["unchunked"]["metrics"]["ttft_p50_ms"]["median"] == 105.0
    vs = agg["chunked"]["vs_baseline"]
    assert vs["ttft_p50_ms"] == {"median_change": 0.15, "better_in": 0, "rounds": 2}
    assert vs["itl_p99_ms"]["better_in"] == 2 and vs["itl_p99_ms"]["median_change"] == pytest.approx(-0.6)
    meta = {"started_at": "t", "git_commit": "c", "gpu": "g", "model": "m", "config": "c.yaml",
            "variants": ["unchunked", "chunked"], "measured_rounds": 2, "warmup_rounds": 1, "seeds": [1, 2, 3],
            "variant_engine": {v: chunked_ab.variant_engine(v, 128, 256) for v in ("unchunked", "chunked")}}
    md = chunked_ab.render_markdown({"meta": meta, "aggregate": {"long_prompt": agg}, "warnings": [], "runs": runs})
    assert "`chunked` | +15.0% (0/2)" in md and "## `long_prompt`" in md


def test_every_variant_is_a_valid_server_config():
    base = {"model": {"name": "m", "weights_dir": "/tmp", "tokenizer_dir": "/tmp"}}
    for variant in chunked_ab.VARIANTS:
        cfg = ServerConfig.model_validate({**base, "engine": chunked_ab.variant_engine(variant, 128, 256)})
        assert cfg.engine.mixed_steps == (variant != "two-pass")
    with pytest.raises(ValueError):
        chunked_ab.variant_engine("fast", 128, 256)


def test_a_run_with_the_wrong_settings_is_refused():
    settings = chunked_ab.variant_engine("chunked", 128, 256)
    chunked_ab.check_run("r", "chunked", {"mixed_steps": 3, "max_step_tokens": 256}, settings)
    with pytest.raises(RuntimeError, match="budget"):
        chunked_ab.check_run("r", "chunked", {"mixed_steps": 3, "max_step_tokens": 300}, settings)
    with pytest.raises(RuntimeError, match="mixed steps off"):
        chunked_ab.check_run("r", "two-pass", {"mixed_steps": 1, "max_step_tokens": 10}, {})
