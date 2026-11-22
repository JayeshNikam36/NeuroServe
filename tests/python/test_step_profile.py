"""The step-profile tool: its analysis, its trace, and a run against an in-process server."""

import json

import httpx
import pytest
import yaml

from neuroserve.bench import step_profile
from neuroserve.server.app import create_app
from neuroserve.server.config import ServerConfig
from neuroserve.server.engine_service import EngineService
from test_api import MODEL_NAME, live_server, make_app, tok  # noqa: F401  (tok is a fixture)


def _record(start, *, prefill=0.0, decode=0.0, host=0.1, inv=0.05, batch=0, prefill_requests=0, prefill_tokens=0):
    r = {"start_ms": start, "housekeeping_ms": 0.02, "schedule_ms": 0.03, "preempt_ms": 0.0, "prefill_ms": prefill,
         "decode_ms": decode, "host_ms": host, "invariants_ms": inv, "prefill_requests": prefill_requests,
         "prefill_tokens": prefill_tokens, "decode_batch": batch}
    r["total_ms"] = sum(r[p] for p in step_profile.PHASES)
    return r


def _profile(records, idle=100.0):
    totals = {p: sum(r[p] for r in records) for p in (*step_profile.PHASES, "total_ms", "prefill_requests",
                                                     "prefill_tokens", "decode_batch")}
    return {"elapsed_ms": totals["total_ms"] + idle, "idle_ms": idle, "steps": len(records),
            "busy_steps": len(records), "totals": totals, "dropped": 0, "records": records}


def test_the_analysis_splits_time_and_groups_decode_by_batch():
    records = [_record(0.0, prefill=40.0, prefill_requests=1, prefill_tokens=400),
               _record(50.0, decode=10.0, batch=1), _record(70.0, decode=10.0, batch=1),
               _record(90.0, decode=12.0, batch=2), _record(110.0, prefill=10.0, decode=12.0, batch=2,
                                                          prefill_requests=1, prefill_tokens=100)]
    a = step_profile.analyse(_profile(records))
    total = sum(r["total_ms"] for r in records)
    assert a["busy_ms"] == round(total, 1) and a["busy_steps"] == 5
    assert a["share"]["prefill_ms"] == round(50.0 / total, 4) and a["share"]["decode_ms"] == round(44.0 / total, 4)
    assert a["gpu_share"] + a["host_share"] + a["share"]["invariants_ms"] == pytest.approx(1.0, abs=1e-3)
    # Steps that also prefilled are left out of the decode table.
    assert [(d["batch"], d["steps"], d["decode_ms_p50"], d["per_sequence_ms"]) for d in a["decode_by_batch"]] == [
        (1, 2, 10.0, 10.0), (2, 1, 12.0, 6.0)]
    assert a["prefill"]["tokens"] == 500 and a["prefill"]["ms_per_token_p50"] == pytest.approx(0.1, abs=1e-4)
    assert a["prefill"]["tokens_per_s"] == 10000


def test_the_trace_has_a_slice_per_step_and_per_phase():
    records = [_record(0.0, decode=2.0, batch=3), _record(5.0, prefill=1.0, prefill_requests=1, prefill_tokens=8)]
    trace = step_profile.chrome_trace(records, "x")
    slices = [e for e in trace["traceEvents"] if e["ph"] == "X"]
    steps = [e for e in slices if e["name"] == "step"]
    assert [s["ts"] for s in steps] == [0.0, 5000.0] and steps[0]["args"]["decode_batch"] == 3
    first = [e for e in slices if e["ts"] < 5000.0 and e["name"] != "step"]
    assert [e["name"] for e in first] == ["housekeeping", "schedule", "decode", "host", "invariants"]
    assert sum(e["dur"] for e in first) == pytest.approx(steps[0]["dur"])
    json.dumps(trace)  # serialisable


class FakeServer:
    tok = None

    def __init__(self, config, out_dir, name):
        cfg = ServerConfig.model_validate(config)
        _, engine, _ = make_app(self.tok)
        self._serving = live_server(create_app(cfg, EngineService(cfg, engine, self.tok, device_name="fake-gpu")))
        self.url = self._serving.__enter__()
        self.config = config

    def wait_ready(self, timeout_s):
        return httpx.get(f"{self.url}/health").json()

    def stop(self):
        self._serving.__exit__(None, None, None)


def test_a_profile_run_against_the_server(tok, tmp_path, monkeypatch):
    FakeServer.tok = tok
    monkeypatch.setattr(step_profile.prefix_ab, "ServerProcess", FakeServer)
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"model": {"name": MODEL_NAME, "weights_dir": "/w", "tokenizer_dir": "/t"}}))
    out = tmp_path / "out"
    assert step_profile.main(["--config", str(base), "--out-dir", str(out), "--scenarios", "decode,prefill",
                              "--sample-interval-s", "0.05"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    # The fake engine reports one 2 ms decode step per request since the last reset: every scenario
    # starts from a reset profile.
    for name in ("decode", "prefill"):
        s = summary["scenarios"][name]
        assert s["succeeded"] == s["requests"] == 32
        assert s["analysis"]["busy_steps"] == 32 and s["analysis"]["share"]["decode_ms"] == 1.0
        assert len((out / f"steps-{name}.csv").read_text().splitlines()) == 33
        assert json.loads((out / f"trace-{name}.json").read_text())["traceEvents"]
    report = (out / "summary.md").read_text()
    assert report.startswith("# Engine step profile") and "| `decode` |" in report
    assert "Decode-only steps by batch size" in report


def test_unknown_scenarios_are_refused(tmp_path):
    with pytest.raises(SystemExit):
        step_profile.main(["--out-dir", str(tmp_path), "--scenarios", "decode,nope"])
