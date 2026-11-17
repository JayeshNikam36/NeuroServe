"""Policy comparison: the balanced run order, the GPU sampler, the statistics, and a full run."""

import itertools
import json
import sys
import time
from collections import Counter

import pytest

from neuroserve.bench import compare
from neuroserve.bench.gpu_monitor import GpuSampler, summarize_samples
from test_api import MODEL_NAME, live_server, make_app, tok  # noqa: F401  (tok is a fixture)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_the_run_order_is_balanced(n):
    items = [f"p{i}" for i in range(n)]
    orders = compare.balanced_orders(items)
    assert len(orders) == (n if n % 2 == 0 else 2 * n)
    assert all(sorted(order) == items for order in orders)
    # Every policy in every position equally often...
    for position in range(n):
        assert set(Counter(order[position] for order in orders).values()) == {len(orders) // n}
    # ...and every policy right after every other one equally often.
    follows = Counter(pair for order in orders for pair in zip(order, order[1:]))
    assert len(follows) == n * (n - 1) and len(set(follows.values())) == 1


class FakeNvml:
    """The parts of pynvml the sampler uses. Temperature is 'not supported'; only the older name of
    the clock-reasons call exists, as on an older driver."""

    NVML_CLOCK_SM, NVML_CLOCK_MEM, NVML_TEMPERATURE_GPU = 1, 2, 0

    class NVMLError(Exception):
        pass

    def __init__(self, fail_init: bool = False) -> None:
        self.fail_init = fail_init
        self.calls = 0
        self.shut_down = False

    def nvmlInit(self):
        if self.fail_init:
            raise self.NVMLError("Driver Not Loaded")

    def nvmlShutdown(self):
        self.shut_down = True

    def nvmlDeviceGetHandleByIndex(self, index):
        return ("gpu", index)

    def nvmlDeviceGetName(self, handle):
        return b"Fake GPU"  # older pynvml returns bytes

    def nvmlDeviceGetClockInfo(self, handle, clock):
        self.calls += 1
        return 2000 if clock == self.NVML_CLOCK_SM else 8000

    def nvmlDeviceGetTemperature(self, handle, sensor):
        raise self.NVMLError("Not Supported")

    def nvmlDeviceGetPowerUsage(self, handle):
        return 45_500  # milliwatts

    def nvmlDeviceGetUtilizationRates(self, handle):
        return type("Utilization", (), {"gpu": 97, "memory": 40})()

    def nvmlDeviceGetCurrentClocksThrottleReasons(self, handle):
        return 0x4 if self.calls % 4 == 0 else 0  # power-capped every other sample


def test_the_gpu_sampler_records_clocks_power_and_reasons():
    nvml = FakeNvml()
    with GpuSampler(device_index=0, interval_s=0.01, nvml=nvml) as sampler:
        time.sleep(0.15)
    trace = sampler.result()
    assert trace["available"] and trace["name"] == "Fake GPU" and trace["reason"] is None and nvml.shut_down
    samples = trace["samples"]
    assert len(samples) >= 4 and samples[0]["t_s"] <= samples[-1]["t_s"]
    s = samples[0]
    assert (s["sm_clock_mhz"], s["mem_clock_mhz"], s["power_w"], s["util_pct"]) == (2000, 8000, 45.5, 97)
    assert s["temp_c"] is None  # unsupported fields are None, not an error
    summary = trace["summary"]
    assert summary["sm_clock_mhz_median"] == 2000 and summary["temp_c_max"] is None
    assert summary["thermal_slowdown_frac"] == 0 and 0.3 <= summary["power_cap_frac"] <= 0.7


def test_without_nvml_the_run_goes_ahead_without_samples(monkeypatch):
    with GpuSampler(nvml=FakeNvml(fail_init=True)) as sampler:
        pass
    trace = sampler.result()
    assert not trace["available"] and "Driver Not Loaded" in trace["reason"] and trace["samples"] == []
    monkeypatch.setitem(sys.modules, "pynvml", None)  # makes "import pynvml" fail
    with GpuSampler() as sampler:
        pass
    assert sampler.result()["reason"] == "nvidia-ml-py is not installed"


def test_throttle_reasons_are_classified():
    samples = [{"sm_clock_mhz": 1800, "reasons": 0x40}, {"sm_clock_mhz": 2000, "reasons": 0x4},
               {"sm_clock_mhz": 2100, "reasons": 0}, {"sm_clock_mhz": 2200, "reasons": 0x8 | 0x4}]
    summary = summarize_samples(samples)
    assert summary["thermal_slowdown_frac"] == 0.5 and summary["power_cap_frac"] == 0.5
    assert summary["sm_clock_mhz_median"] == 2050 and summary["sm_clock_mhz_min"] == 1800


def _run(round_, policy, ttft_p95, measured=True, clock=2000.0, thermal=0.0):
    metrics = {key: 10.0 for key, _, _ in compare.METRICS}
    metrics.update(ttft_p95_ms=ttft_p95, failed=0)
    gpu = {"available": True, "reason": None,
           "summary": {"sm_clock_mhz_median": clock, "sm_clock_mhz_min": clock, "temp_c_max": 70,
                       "power_w_mean": 50.0, "util_pct_mean": 90.0, "thermal_slowdown_frac": thermal,
                       "power_cap_frac": 0.5}}
    return {"round": round_, "measured": measured, "profile": "short", "policy": policy, "metrics": metrics,
            "engine": {"preemptions": 0}, "gpu": gpu}


def test_policies_are_compared_round_by_round():
    runs = [_run(1, "fcfs", 999.0, measured=False), _run(1, "decode-first", 1.0, measured=False)]  # warm-up
    for round_, value in ((2, 80.0), (3, 90.0), (4, 120.0)):
        runs += [_run(round_, "fcfs", 100.0), _run(round_, "decode-first", value)]
    agg = compare.aggregate(runs, baseline="fcfs")["short"]
    assert agg["fcfs"]["rounds"] == 3 and agg["fcfs"]["vs_baseline"] == {}  # the warm-up round is excluded
    assert agg["decode-first"]["metrics"]["ttft_p95_ms"] == {"median": 90.0, "min": 80.0, "max": 120.0}
    # Per-round ratios 0.8, 0.9, 1.2: median change -10%; lower TTFT is better in 2 of 3 rounds.
    assert agg["decode-first"]["vs_baseline"]["ttft_p95_ms"] == {"median_change": -0.1, "better_in": 2, "rounds": 3}
    # Equal throughput is not "better": higher is better for it, and it was never higher.
    assert agg["decode-first"]["vs_baseline"]["output_tokens_per_s"]["better_in"] == 0


def test_throttled_or_off_clock_runs_are_flagged():
    runs = [_run(2, "fcfs", 1.0), _run(2, "decode-first", 1.0, clock=1700.0),
            _run(3, "fcfs", 1.0, thermal=0.25), _run(3, "decode-first", 1.0)]
    warnings = compare.gpu_warnings(runs)
    assert len(warnings) == 2
    assert "round 2 short decode-first: median SM clock 1700 MHz" in warnings[0]
    assert "round 3 short fcfs: thermal slowdown in 25% of samples" == warnings[1]


def test_a_comparison_against_the_server(tok, tmp_path):
    app, engine, _ = make_app(tok)
    with live_server(app) as base_url:
        code = compare.main(["--url", base_url, "--policies", "fcfs,decode-first", "--profiles", "short",
                             "--requests", "4", "--rate", "50", "--rounds", "2", "--warmup-rounds", "1",
                             "--cooldown-s", "0", "--sample-interval-s", "0.05", "--out-dir", str(tmp_path)])
    assert code == 0
    # Warm-up round (first order), then the two balanced orders; then the original policy is restored.
    assert engine.policy_history == ["fcfs", "decode-first", "fcfs", "decode-first", "decode-first", "fcfs",
                                     "fcfs"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    runs = summary["runs"]
    assert [(r["round"], r["policy"], r["measured"]) for r in runs] == [
        (1, "fcfs", False), (1, "decode-first", False), (2, "fcfs", True), (2, "decode-first", True),
        (3, "decode-first", True), (3, "fcfs", True)]
    for round_, pair in itertools.groupby(runs, key=lambda r: r["round"]):
        pair = list(pair)
        assert pair[0]["schedule_sha256"] == pair[1]["schedule_sha256"]   # same requests within a round
        assert pair[0]["seed"] == round_                                   # seed 1 + (round - 1)
    assert runs[0]["schedule_sha256"] != runs[2]["schedule_sha256"]        # new requests each round
    assert all(r["metrics"]["succeeded"] == 4 and r["engine"]["completed"] == 4 for r in runs)
    assert all(r["decisions"] == {"count": 4, "dropped": 0} for r in runs)  # one admission per request
    agg = summary["aggregate"]["short"]
    assert agg["fcfs"]["rounds"] == 2 and agg["decode-first"]["vs_baseline"]["ttft_p95_ms"]["rounds"] == 2
    assert summary["meta"]["model"] == MODEL_NAME and summary["meta"]["orders"] == [
        ["fcfs", "decode-first"], ["decode-first", "fcfs"]]
    assert len(list((tmp_path / "runs").glob("r0*-short-*.json"))) == 6
    report = (tmp_path / "summary.md").read_text()
    assert report.startswith("# Scheduling policies compared") and "| `decode-first` |" in report


def test_a_run_that_did_not_use_its_policy_stops_the_comparison(tok, tmp_path):
    app, engine, _ = make_app(tok)
    engine.set_scheduling_policy = lambda name: None  # accepts every policy, never switches
    with live_server(app) as base_url:
        with pytest.raises(RuntimeError, match=r"a run under 'decode-first': the decision log shows \['fcfs'\]"):
            compare.main(["--url", base_url, "--policies", "fcfs,decode-first", "--profiles", "short",
                          "--requests", "3", "--rate", "50", "--rounds", "1", "--warmup-rounds", "0",
                          "--cooldown-s", "0", "--out-dir", str(tmp_path)])
