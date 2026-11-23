"""API tests: OpenAI format, validation, errors (fake engine) and one real end-to-end run."""

import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

import neuroserve
from neuroserve.server.app import create_app
from neuroserve.server.config import ServerConfig
from neuroserve.server.engine_service import EngineService
from neuroserve.tokenizer import ChatTokenizer

TOKENIZER_DIR = Path(os.environ.get("NS_TOKENIZER_DIR", "/models/llama-3.2-1b-instruct"))
MODEL_DIR = Path(os.environ.get("NS_MODEL_DIR", "/models/llama-3.2-1b-fp16"))
MODEL_NAME = "llama-3.2-1b"
REPLY = "Paris is the capital of France."

pytestmark = pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer.json").exists(),
                                reason=f"tokenizer not found in {TOKENIZER_DIR}")


def make_config(**engine) -> ServerConfig:
    return ServerConfig.model_validate({
        "model": {"name": MODEL_NAME, "weights_dir": str(MODEL_DIR), "tokenizer_dir": str(TOKENIZER_DIR)},
        "engine": {"max_prompt_tokens": 128, **engine},
        "limits": {"default_max_tokens": 16, "max_tokens_limit": 64},
    })


def test_prompt_limit_must_fit_the_token_budget():
    with pytest.raises(ValueError, match="max_batch_tokens"):
        make_config(max_prompt_tokens=600, max_batch_tokens=512)


def test_with_chunked_prefill_a_prompt_may_be_longer_than_a_step():
    config = make_config(max_prompt_tokens=600, max_batch_tokens=512, prefill_chunk_tokens=256)
    assert config.engine.prefill_chunk_tokens == 256
    with pytest.raises(ValueError):
        make_config(prefill_chunk_tokens=-1)


class FakeStream:
    """Like neuroserve.TokenStream: hands out the tokens one per read(), then the result.

    `gate`: when set, generation pauses after the first token (and result() waits) until the gate
    is set or the request is cancelled. `error`: raised by result(), after the tokens.
    """

    def __init__(self, out, gate: threading.Event | None = None, error: Exception | None = None) -> None:
        self.request_id = out.request_id
        self._out = out
        self._pending = list(out.tokens)
        self._gate = gate
        self._error = error
        self._sent = 0
        self.cancelled = threading.Event()
        self.finished = False

    def _gate_open(self, timeout: float | None) -> bool:
        """Waits up to `timeout` s for the gate or a cancellation; None: up to 10 s, then fails."""
        deadline = time.monotonic() + (10 if timeout is None else timeout)
        while not (self._gate.is_set() or self.cancelled.is_set()):
            if time.monotonic() >= deadline:
                assert timeout is not None, "the gate was never opened (did the client get the first token?)"
                return False
            time.sleep(0.005)
        return True

    def read(self, timeout=None):
        if self._sent == 1 and self._gate is not None and not self._gate_open(timeout):
            return []  # like the engine: nothing new within the timeout
        if not self._pending or self.cancelled.is_set():
            return None
        self._sent += 1
        return [self._pending.pop(0)]

    def result(self):
        try:
            if self._gate is not None:
                self._gate_open(None)
            if self.cancelled.is_set():
                raise neuroserve.RequestCancelled("request cancelled")
            if self._error is not None:
                raise self._error
            return self._out
        finally:
            self.finished = True


class FakeEngine:
    """Like the real engine's run(): a fixed reply, then filler, ending at the first stop token."""

    def __init__(self, reply_ids: list[int], filler: int, error: Exception | None = None,
                 stream_error: Exception | None = None, gate: threading.Event | None = None) -> None:
        self.reply_ids = reply_ids
        self.filler = filler
        self.error = error
        self.stream_error = stream_error  # streaming: raised after the tokens, when the result is read
        self.gate = gate
        self.streams: dict[str, FakeStream] = {}
        self.cancelled: list[str] = []  # engine request ids cancelled, in order
        self.calls: list[tuple[list[int], int]] = []
        self.stop_token_ids: list[list[int]] = []
        self.client_request_ids: list[str] = []
        self.priorities: list[int] = []
        self.policy = "fcfs"
        self.next_policy: str | None = None  # like the real engine: applied at the next step
        self.policy_history: list[str] = []  # every set_scheduling_policy call, in order
        self.profile_reset_at = 0  # len(calls) at the last reset_step_profile()
        self.decision_log = [{"seq": i, "step": i, "time_ms": 0, "request_id": f"ns-{i:06d}",
                              "client_request_id": "", "action": "admit", "reason": "admitted", "policy": "fcfs",
                              "numbers": {"free_pages": 64}} for i in range(1, 6)]

    def run(self, prompt_ids, max_new_tokens, *, stop_token_ids=(), client_request_id="", priority=1):
        if self.error is not None:
            raise self.error
        if self.next_policy is not None:  # this request's step is the engine's next step
            self.policy, self.next_policy = self.next_policy, None
        self.calls.append((list(prompt_ids), max_new_tokens))
        seq = self.decision_log[-1]["seq"] + 1
        self.decision_log.append({"seq": seq, "step": seq, "time_ms": 0, "request_id": f"ns-{len(self.calls):06d}",
                                  "client_request_id": client_request_id, "action": "admit", "reason": "admitted",
                                  "policy": self.policy, "numbers": {"free_pages": 64}})
        self.stop_token_ids.append(list(stop_token_ids))
        self.client_request_ids.append(client_request_id)
        self.priorities.append(priority)
        out = []
        for token in (self.reply_ids + [self.filler] * max_new_tokens)[:max_new_tokens]:
            out.append(token)
            if token in stop_token_ids:
                break
        finish = "stop" if out[-1] in stop_token_ids else "length"
        return SimpleNamespace(request_id=f"ns-{len(self.calls):06d}", tokens=out, finish_reason=finish,
                               queue_ms=0.1, ttft_ms=0.5, e2e_ms=1.0, itl_ms=0.05, max_itl_ms=0.08)

    def stream(self, prompt_ids, max_new_tokens, *, stop_token_ids=(), client_request_id="", priority=1):
        out = self.run(prompt_ids, max_new_tokens, stop_token_ids=stop_token_ids,
                       client_request_id=client_request_id, priority=priority)
        stream = FakeStream(out, gate=self.gate, error=self.stream_error)
        self.streams[stream.request_id] = stream
        return stream

    def cancel(self, request_id):
        stream = self.streams.get(request_id)
        if stream is None or stream.finished or stream.cancelled.is_set():
            return False
        stream.cancelled.set()
        self.cancelled.append(request_id)
        return True

    def kv_stats(self):
        return {"page_size_tokens": 16, "total_pages": 64, "free_pages": 64, "max_pages_per_sequence": 64,
                "shared_pages": 0, "indexed_pages": 0, "prefix_lookups": 0, "prefix_lookup_pages": 0,
                "prefix_hit_pages": 0, "cached_pages": 0, "evicted_pages": 0, "prefix_reused_tokens": 0}

    def stats(self):
        return {"submitted": len(self.calls), "completed": len(self.calls), "pending": 0, "active": 0,
                "policy": self.policy, "preemptions": 0}

    def set_scheduling_policy(self, name):
        if name not in neuroserve.scheduling_policies:
            raise ValueError(f"unknown scheduling policy '{name}'")
        self.next_policy = name
        self.policy_history.append(name)

    def latency(self):
        summary = {"count": len(self.calls), "mean": 1.0, "p50": 1.0, "p95": 1.0, "p99": 1.0, "max": 1.0}
        return {name: dict(summary) for name in ("queue_ms", "ttft_ms", "itl_ms", "e2e_ms")}

    def decisions(self, since=0, limit=100):
        page = [d for d in self.decision_log if d["seq"] > since][:limit]
        return {"data": page, "next_since": page[-1]["seq"] if page else since, "dropped": 0}

    def step_profile(self, records=False):
        # Like the real engine: one record per call since the last reset, 2 ms of decode each.
        steps = len(self.calls) - self.profile_reset_at
        step = {"housekeeping_ms": 0.0, "schedule_ms": 0.0, "preempt_ms": 0.0, "prefill_ms": 0.0,
                "decode_ms": 2.0, "host_ms": 0.0, "invariants_ms": 0.0, "total_ms": 2.0,
                "prefill_requests": 0, "prefill_tokens": 0, "decode_batch": 1}
        out = {"elapsed_ms": 10.0 * steps, "idle_ms": 8.0 * steps, "steps": steps, "busy_steps": steps,
               "totals": {k: v * steps for k, v in step.items()}, "dropped": 0}
        if records:
            out["records"] = [{**step, "start_ms": 10.0 * i} for i in range(steps)]
        return out

    def reset_step_profile(self):
        self.profile_reset_at = len(self.calls)


@pytest.fixture(scope="module")
def tok() -> ChatTokenizer:
    return ChatTokenizer(TOKENIZER_DIR)


def make_app(tok, *, stop=True, error=None, stream_error=None, gate=None):
    eot = tok.tokenizer.token_to_id("<|eot_id|>")
    reply_ids = tok.encode(REPLY, add_special_tokens=False) + ([eot] if stop else [])
    filler = tok.encode(" filler", add_special_tokens=False)[0]
    engine = FakeEngine(reply_ids, filler, error, stream_error, gate)
    config = make_config()
    return create_app(config, EngineService(config, engine, tok, device_name="fake-gpu")), engine, reply_ids


def chat_body(**overrides) -> dict:
    body = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "What is the capital of France?"}]}
    body.update(overrides)
    return body


def test_health_and_models(tok):
    app, _, _ = make_app(tok)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["model"] == MODEL_NAME and health["gpu"] == "fake-gpu"
        assert set(health["latency"]) == {"queue_ms", "ttft_ms", "itl_ms", "e2e_ms"}
        models = client.get("/v1/models").json()
        assert models["object"] == "list" and models["data"][0]["id"] == MODEL_NAME


def test_chat_completion_has_openai_format(tok):
    app, engine, reply_ids = make_app(tok)
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=chat_body())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
    assert body["model"] == MODEL_NAME
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": REPLY}
    assert choice["finish_reason"] == "stop"
    prompt_len = len(tok.encode_chat(chat_body()["messages"]))
    assert body["usage"] == {"prompt_tokens": prompt_len, "completion_tokens": len(reply_ids),
                             "total_tokens": prompt_len + len(reply_ids)}
    assert engine.calls[0][1] == 16  # default_max_tokens from the config


def test_no_stop_token_reports_length(tok):
    app, _, _ = make_app(tok, stop=False)
    with TestClient(app) as client:
        body = client.post("/v1/chat/completions", json=chat_body(max_tokens=4)).json()
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 4


def test_max_completion_tokens_takes_precedence(tok):
    app, engine, _ = make_app(tok)
    with TestClient(app) as client:
        client.post("/v1/chat/completions", json=chat_body(max_completion_tokens=5, max_tokens=9))
    assert engine.calls[0][1] == 5


def test_temperature_zero_is_accepted(tok):
    app, _, _ = make_app(tok)
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=chat_body(temperature=0)).status_code == 200


@pytest.mark.parametrize("overrides, status, param", [
    ({"model": "gpt-4"}, 404, "model"),
    ({"stream_options": {"include_usage": True}}, 400, "stream_options"),  # only with stream=true
    ({"temperature": 0.7}, 400, "temperature"),
    ({"n": 2}, 400, "n"),
    ({"max_tokens": 1000}, 400, "max_tokens"),
    ({"max_tokens": 0}, 400, "max_tokens"),
    ({"messages": []}, 400, "messages"),
    ({"messages": [{"role": "robot", "content": "hi"}]}, 400, "messages.0.role"),
    ({"neuroserve": {"priority": 3}}, 400, "neuroserve.priority"),
    ({"neuroserve": {"priority": -1}}, 400, "neuroserve.priority"),
    ({"max_tokens": "ten"}, 400, "max_tokens"),
    ({"temperature": -1}, 400, "temperature"),
    ({"top_p": 0}, 400, "top_p"),
    ({"neuroserve": {"no_such_option": True}}, 400, "neuroserve.no_such_option"),
])
def test_bad_requests_get_openai_errors(tok, overrides, status, param):
    app, engine, _ = make_app(tok)
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=chat_body(**overrides))
    assert resp.status_code == status, resp.text
    error = resp.json()["error"]
    assert error["param"] == param and error["message"] and error["type"]
    assert engine.calls == []  # rejected before reaching the engine


def test_missing_fields_are_400(tok):
    app, _, _ = make_app(tok)
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={})
    assert resp.status_code == 400 and resp.json()["error"]["type"] == "invalid_request_error"


def test_prompt_too_long_is_413(tok):
    app, engine, _ = make_app(tok)
    long_body = chat_body(messages=[{"role": "user", "content": "word " * 400}])
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=long_body)
    assert resp.status_code == 413 and resp.json()["error"]["code"] == "context_length_exceeded"
    assert engine.calls == []


def test_kv_exhaustion_is_503(tok):
    app, _, _ = make_app(tok, error=neuroserve.KvCacheExhausted("no free pages"))
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=chat_body())
    assert resp.status_code == 503 and resp.json()["error"]["code"] == "kv_cache_exhausted"


@pytest.mark.skipif(not (MODEL_DIR / "weights.bin").exists(), reason=f"model not found in {MODEL_DIR}")
def test_real_engine_end_to_end():
    with TestClient(create_app(make_config(num_pages=512))) as client:
        resp = client.post("/v1/chat/completions", json=chat_body(max_tokens=16))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["choices"][0]["finish_reason"] in ("stop", "length")
        assert 1 <= body["usage"]["completion_tokens"] <= 16
        kv = client.get("/health").json()["kv_cache"]
        assert kv["free_pages"] == kv["total_pages"], "KV pages leaked after a request"


# ---- Day 8: /v1/completions and NeuroServe options ----------------------------------------------

def test_completions_with_text_prompt(tok):
    app, engine, _ = make_app(tok)
    with TestClient(app) as client:
        resp = client.post("/v1/completions",
                           json={"model": MODEL_NAME, "prompt": "The capital of France is", "max_tokens": 16})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["object"] == "text_completion" and body["id"].startswith("cmpl-")
    assert body["choices"][0]["text"] == REPLY and body["choices"][0]["finish_reason"] == "stop"
    assert engine.calls[0][0] == tok.encode("The capital of France is")  # text prompts get <|begin_of_text|>
    assert body["neuroserve"]["completion_token_ids"] is None             # only on request


def test_completions_token_ids_are_used_verbatim(tok):
    app, engine, _ = make_app(tok)
    prompt = [128000, 791, 6864, 315]
    with TestClient(app) as client:
        body = client.post("/v1/completions", json={"model": MODEL_NAME, "prompt": prompt}).json()
    assert engine.calls[0][0] == prompt
    assert body["usage"]["prompt_tokens"] == len(prompt)


def test_return_token_ids_includes_the_stop_token(tok):
    app, _, reply_ids = make_app(tok)
    with TestClient(app) as client:
        body = client.post("/v1/chat/completions",
                           json=chat_body(neuroserve={"return_token_ids": True})).json()
    assert body["neuroserve"]["completion_token_ids"] == reply_ids  # reply + <|eot_id|>
    assert body["usage"]["completion_tokens"] == len(reply_ids)


def test_ignore_eos_generates_exactly_max_tokens(tok):
    app, _, _ = make_app(tok)
    with TestClient(app) as client:
        body = client.post("/v1/chat/completions",
                           json=chat_body(max_tokens=20, neuroserve={"ignore_eos": True, "return_token_ids": True})).json()
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 20 == len(body["neuroserve"]["completion_token_ids"])


def test_stop_tokens_and_request_id_are_passed_to_the_engine(tok):
    app, engine, _ = make_app(tok)
    with TestClient(app) as client:
        client.post("/v1/chat/completions", json=chat_body(), headers={"X-Request-ID": "trace-42"})
        client.post("/v1/chat/completions", json=chat_body(neuroserve={"ignore_eos": True}))
    assert set(engine.stop_token_ids[0]) == set(tok.stop_token_ids)  # the engine stops, not Python
    assert engine.stop_token_ids[1] == []                             # ignore_eos: no stop tokens
    assert engine.client_request_ids[0] == "trace-42"                 # links engine logs to the request


def test_priority_is_passed_to_the_engine(tok):
    app, engine, _ = make_app(tok)
    with TestClient(app) as client:
        client.post("/v1/chat/completions", json=chat_body())
        client.post("/v1/chat/completions", json=chat_body(neuroserve={"priority": 2}))
        client.post("/v1/completions", json={"model": MODEL_NAME, "prompt": "hi", "neuroserve": {"priority": 0}})
    assert engine.priorities == [1, 2, 0]  # normal by default


@pytest.mark.parametrize("payload, param", [
    ({"prompt": ""}, "prompt"),
    ({"prompt": []}, "prompt"),
    ({"prompt": [1, 999999]}, "prompt"),                       # outside the vocabulary
    ({"prompt": "hi", "neuroserve": {"bogus": True}}, "neuroserve.bogus"),
])
def test_bad_completion_requests_are_400(tok, payload, param):
    app, engine, _ = make_app(tok)
    with TestClient(app) as client:
        resp = client.post("/v1/completions", json={"model": MODEL_NAME, **payload})
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["param"] == param
    assert engine.calls == []


# ---- Day 18: streaming over server-sent events ---------------------------------------------------

def sse_events(resp) -> list:
    """The data of each server-sent event: parsed JSON, or the final "[DONE]"."""
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = []
    for line in resp.text.split("\n"):
        if line.startswith("data: "):
            data = line[len("data: "):]
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


def test_chat_stream_sends_openai_chunks(tok):
    app, _, reply_ids = make_app(tok)
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=chat_body(stream=True))
    assert resp.status_code == 200, resp.text
    events = sse_events(resp)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert all(c["object"] == "chat.completion.chunk" and c["id"] == chunks[0]["id"] for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == REPLY                                       # the same text as without streaming
    assert len(chunks) >= 4                                    # role, several pieces of text, finish
    last = chunks[-1]
    assert last["choices"][0] == {"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}
    assert last["neuroserve"]["generated_tokens"] == len(reply_ids)
    assert "usage" not in last                                 # only with stream_options.include_usage


def test_stream_with_include_usage_ends_with_a_usage_chunk(tok):
    app, _, reply_ids = make_app(tok)
    with TestClient(app) as client:
        body = chat_body(stream=True, stream_options={"include_usage": True}, neuroserve={"return_token_ids": True})
        events = sse_events(client.post("/v1/chat/completions", json=body))
        plain = client.post("/v1/chat/completions", json=chat_body()).json()
    usage_chunk = events[-2]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"] == plain["usage"]              # the same numbers as without streaming
    assert usage_chunk["neuroserve"]["completion_token_ids"] == reply_ids
    assert events[-3]["choices"][0]["finish_reason"] == "stop"


def test_completions_stream(tok):
    app, _, _ = make_app(tok)
    with TestClient(app) as client:
        events = sse_events(client.post("/v1/completions",
                                        json={"model": MODEL_NAME, "prompt": "The capital of France is",
                                              "stream": True}))
    chunks = events[:-1]
    assert all(c["object"] == "text_completion" for c in chunks)
    assert "".join(c["choices"][0]["text"] for c in chunks) == REPLY
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop" and events[-1] == "[DONE]"


def test_stream_errors_before_the_first_token_are_http_errors(tok):
    app, engine, _ = make_app(tok)
    long_body = chat_body(stream=True, messages=[{"role": "user", "content": "word " * 400}])
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=long_body)
    assert resp.status_code == 413 and resp.json()["error"]["code"] == "context_length_exceeded"
    assert engine.calls == []


def test_an_error_during_the_stream_is_sent_as_an_error_event(tok):
    app, _, _ = make_app(tok, stream_error=neuroserve.EngineStopped("engine stopping"))
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=chat_body(stream=True))
    assert resp.status_code == 200  # already sent when the error happened
    events = sse_events(resp)
    assert events[-1]["error"]["code"] == "service_unavailable"
    assert "[DONE]" not in events


def test_the_official_openai_client_streams(tok):
    from openai import OpenAI

    app, _, reply_ids = make_app(tok)
    with TestClient(app) as http:
        client = OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=http)
        chunks = list(client.chat.completions.create(
            model=MODEL_NAME, messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=16, stream=True, stream_options={"include_usage": True}))
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == REPLY
    assert chunks[-1].usage.completion_tokens == len(reply_ids)
    assert chunks[-1].neuroserve["generated_tokens"] == len(reply_ids)  # NeuroServe's extension field


@contextmanager
def live_server(app):
    """Serves the app with uvicorn on a free local port (for tests that need a real connection)."""
    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "server did not start"
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_tokens_reach_the_client_before_generation_ends(tok):
    """Through a real HTTP server: the first token must arrive while the engine is still
    generating (nothing on the way, such as a middleware, may buffer the stream)."""
    from openai import OpenAI

    gate = threading.Event()  # the fake engine produces nothing after the first token until this is set
    app, _, _ = make_app(tok, gate=gate)
    with live_server(app) as base_url:
        client = OpenAI(base_url=f"{base_url}/v1", api_key="unused", timeout=10)
        stream = client.chat.completions.create(
            model=MODEL_NAME, messages=[{"role": "user", "content": "hi"}], max_tokens=16, stream=True)
        pieces = []
        for chunk in stream:
            content = chunk.choices[0].delta.content if chunk.choices else None
            if content:
                pieces.append(content)
                assert gate.is_set() or len(pieces) == 1  # the first piece came while generation was paused
                gate.set()
        assert "".join(pieces) == REPLY
        gate.set()


# ---- Day 19: cancellation ------------------------------------------------------------------------

def wait_until(condition, what: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


def read_until_first_text(lines) -> None:
    for line in lines:
        if line.startswith("data: ") and json.loads(line[6:])["choices"][0]["delta"].get("content"):
            return
    raise AssertionError("the stream ended before any text")


def test_delete_of_an_unknown_request_is_404(tok):
    app, _, _ = make_app(tok)
    with TestClient(app) as client:
        resp = client.delete("/v1/requests/no-such-request")
    assert resp.status_code == 404 and resp.json()["error"]["code"] == "request_not_found"


def test_delete_cancels_a_streaming_request(tok):
    gate = threading.Event()  # never set: the request would run until cancelled
    app, engine, _ = make_app(tok, gate=gate)
    with live_server(app) as base:
        with httpx.stream("POST", f"{base}/v1/chat/completions", json=chat_body(stream=True),
                          headers={"X-Request-ID": "job-1"}, timeout=10) as resp:
            lines = resp.iter_lines()
            read_until_first_text(lines)
            cancel = httpx.delete(f"{base}/v1/requests/job-1", timeout=10)
            assert cancel.status_code == 200, cancel.text
            assert cancel.json() == {"id": "job-1", "object": "request.cancellation", "cancelled": True}
            events = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
        assert events[-1]["error"]["code"] == "request_cancelled"  # the stream ends with the reason
        assert engine.cancelled == ["ns-000001"]
        assert httpx.delete(f"{base}/v1/requests/job-1", timeout=10).status_code == 404  # no longer in flight


def test_delete_cancels_a_non_streaming_request(tok):
    gate = threading.Event()
    app, engine, _ = make_app(tok, gate=gate)
    responses = []
    with live_server(app) as base:
        caller = threading.Thread(target=lambda: responses.append(httpx.post(
            f"{base}/v1/chat/completions", json=chat_body(), headers={"X-Request-ID": "job-2"}, timeout=10)))
        caller.start()
        wait_until(lambda: httpx.delete(f"{base}/v1/requests/job-2", timeout=10).status_code == 200,
                   "the request to become cancellable")
        caller.join(timeout=10)
    assert responses[0].status_code == 499
    assert responses[0].json()["error"]["code"] == "request_cancelled"
    assert engine.cancelled == ["ns-000001"]


def test_client_disconnect_cancels_a_stream(tok):
    gate = threading.Event()
    app, engine, _ = make_app(tok, gate=gate)
    with live_server(app) as base:
        with httpx.stream("POST", f"{base}/v1/chat/completions", json=chat_body(stream=True), timeout=10) as resp:
            read_until_first_text(resp.iter_lines())
        # Leaving the block closes the connection mid-stream.
        wait_until(lambda: engine.cancelled == ["ns-000001"], "the engine request to be cancelled")


def test_client_disconnect_cancels_a_non_streaming_request(tok):
    gate = threading.Event()
    app, engine, _ = make_app(tok, gate=gate)
    with live_server(app) as base:
        with pytest.raises(httpx.ReadTimeout):  # the client gives up and closes the connection
            httpx.post(f"{base}/v1/chat/completions", json=chat_body(), timeout=0.5)
        wait_until(lambda: engine.cancelled == ["ns-000001"], "the engine request to be cancelled")
        # The server forgets the request once it has finished (with its 499), a moment after the cancel.
        wait_until(lambda: app.state.in_flight == {}, "the server to forget the cancelled request")


# ---- Day 20: input limits, queue limits, validation ----------------------------------------------

def test_malformed_json_is_400(tok):
    app, engine, _ = make_app(tok)
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", content=b'{"model": "x", "messages": [',
                           headers={"Content-Type": "application/json"})
    assert resp.status_code == 400 and resp.json()["error"]["type"] == "invalid_request_error"
    assert engine.calls == []


def test_completions_prompt_over_the_limit_is_413(tok):
    app, engine, _ = make_app(tok)
    with TestClient(app) as client:
        resp = client.post("/v1/completions", json={"model": MODEL_NAME, "prompt": [1] * 129})  # limit is 128
    assert resp.status_code == 413
    assert resp.json()["error"] == {"message": resp.json()["error"]["message"], "type": "invalid_request_error",
                                    "param": "prompt", "code": "context_length_exceeded"}
    assert engine.calls == []


@pytest.mark.parametrize("stream", [False, True])
def test_a_full_queue_is_429_with_retry_after(tok, stream):
    app, _, _ = make_app(tok, error=neuroserve.QueueFull("the queue is full: 256 requests are waiting"))
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=chat_body(stream=stream))
    assert resp.status_code == 429  # before any stream starts: submit is refused at once
    assert resp.headers["Retry-After"] == "1"
    assert resp.json()["error"]["code"] == "queue_full"


def test_a_queue_timeout_is_503_with_retry_after(tok):
    app, _, _ = make_app(tok, stream_error=neuroserve.QueueTimeout("the request waited 30001 ms in the queue"))
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=chat_body())
    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "1"
    assert resp.json()["error"]["code"] == "queue_timeout"


def test_a_queue_timeout_ends_a_stream_with_an_error_event(tok):
    app, _, _ = make_app(tok, stream_error=neuroserve.QueueTimeout("the request waited 30001 ms in the queue"))
    with TestClient(app) as client:
        events = sse_events(client.post("/v1/chat/completions", json=chat_body(stream=True)))
    assert events[-1]["error"]["code"] == "queue_timeout"


@pytest.mark.parametrize("engine_overrides", [{"max_queue": 0}, {"queue_timeout_ms": -1},
                                              {"decision_log_capacity": 0}])
def test_bad_queue_limits_in_the_config_are_refused(engine_overrides):
    with pytest.raises(ValueError):
        make_config(**engine_overrides)


# ---- Day 21: latency instrumentation -------------------------------------------------------------

def test_every_response_carries_its_timings(tok):
    app, _, _ = make_app(tok)
    with TestClient(app) as client:
        plain = client.post("/v1/chat/completions", json=chat_body()).json()["neuroserve"]
        events = sse_events(client.post("/v1/chat/completions", json=chat_body(stream=True)))
    streamed = events[-2]["neuroserve"]  # the last chunk before [DONE]
    for info in (plain, streamed):
        assert info["queue_ms"] == 0.1 and info["ttft_ms"] == 0.5 and info["e2e_ms"] == 1.0
        assert info["itl_ms"] == 0.05 and info["max_itl_ms"] == 0.08


# ---- Day 22: explainable scheduling decisions ----------------------------------------------------

def test_decisions_endpoint_pages_with_a_cursor(tok):
    app, _, _ = make_app(tok)
    with TestClient(app) as client:
        first = client.get("/v1/runtime/decisions", params={"limit": 2}).json()
        rest = client.get("/v1/runtime/decisions", params={"since": first["next_since"]}).json()
    assert first["object"] == "list" and [d["seq"] for d in first["data"]] == [1, 2]
    assert first["data"][0]["numbers"] == {"free_pages": 64}
    assert [d["seq"] for d in rest["data"]] == [3, 4, 5] and rest["next_since"] == 5 and rest["dropped"] == 0


@pytest.mark.parametrize("params, param", [({"limit": 0}, "query.limit"), ({"limit": 5000}, "query.limit"),
                                           ({"since": -1}, "query.since")])
def test_decisions_endpoint_validates_its_parameters(tok, params, param):
    app, _, _ = make_app(tok)
    with TestClient(app) as client:
        resp = client.get("/v1/runtime/decisions", params=params)
    assert resp.status_code == 400 and resp.json()["error"]["param"] == param


# ---- Day 24: switching the scheduling policy at runtime ------------------------------------------

def test_the_policy_switches_at_the_next_engine_step(tok):
    app, engine, _ = make_app(tok)
    with TestClient(app) as client:
        before = client.get("/v1/runtime/policy").json()
        requested = client.put("/v1/runtime/policy", json={"policy": "decode-first"}).json()
        idle = client.get("/health").json()["engine"]["policy"]
        client.post("/v1/chat/completions", json=chat_body(max_tokens=4))
        after = client.get("/v1/runtime/policy").json()
    assert before == {"object": "runtime.policy", "policy": "fcfs", "pending": None,
                      "available": list(neuroserve.scheduling_policies)}
    # An idle engine takes no steps: the new policy is pending until a request arrives.
    assert (requested["policy"], requested["pending"]) == ("fcfs", "decode-first") and idle == "fcfs"
    assert (after["policy"], after["pending"]) == ("decode-first", None)
    assert engine.decision_log[-1]["policy"] == "decode-first" and engine.policy_history == ["decode-first"]


@pytest.mark.parametrize("payload, code", [({"policy": "lifo"}, "unknown_policy"), ({}, None),
                                           ({"policy": "fcfs", "extra": 1}, None)])
def test_a_bad_policy_change_is_400_and_changes_nothing(tok, payload, code):
    app, engine, _ = make_app(tok)
    with TestClient(app) as client:
        resp = client.put("/v1/runtime/policy", json=payload)
        state = client.get("/v1/runtime/policy").json()
        policy = {"policy": state["policy"], "pending": state["pending"]}
    assert resp.status_code == 400 and resp.json()["error"]["code"] == code
    assert policy == {"policy": "fcfs", "pending": None} and engine.policy_history == []


def test_kv_settings_have_safe_defaults_and_are_checked():
    engine = make_config().engine
    assert engine.prefix_caching is False and engine.kv_admission == "on-demand"
    assert make_config(prefix_caching=True, kv_admission="worst-case").engine.kv_admission == "worst-case"
    with pytest.raises(ValueError):
        make_config(kv_admission="greedy")


def test_runtime_state_reports_the_kv_cache_metrics(tok):
    app, engine, _ = make_app(tok)
    # 64 pages: 10 held by live requests, 6 CACHED, 48 FREE; 2 lookups could reuse 8 pages, 5 hit.
    engine.kv_stats = lambda: {
        "page_size_tokens": 16, "total_pages": 64, "free_pages": 54, "max_pages_per_sequence": 64,
        "shared_pages": 3, "indexed_pages": 9, "prefix_lookups": 2, "prefix_lookup_pages": 8,
        "prefix_hit_pages": 5, "cached_pages": 6, "evicted_pages": 4, "prefix_reused_tokens": 79}
    base_stats = engine.stats
    engine.stats = lambda: {**base_stats(), "preemptions": 7}
    with TestClient(app) as client:
        state = client.get("/v1/runtime/state").json()
    assert state["object"] == "runtime.state"
    assert state["kv_cache"] == {"page_size_tokens": 16, "total_pages": 64, "used_pages": 10, "cached_pages": 6,
                                 "free_pages": 48, "occupancy": 0.1562, "shared_pages": 3, "indexed_pages": 9,
                                 "evicted_pages": 4}
    assert state["prefix_cache"] == {"enabled": False, "lookups": 2, "lookup_pages": 8, "hit_pages": 5,
                                     "miss_pages": 3, "hit_rate": 0.625, "reused_tokens": 79}
    assert state["admission"] == "on-demand" and state["preemptions"] == 7 and state["policy"] == "fcfs"


def test_runtime_state_of_an_idle_engine(tok):
    app, _, _ = make_app(tok)
    with TestClient(app) as client:
        state = client.get("/v1/runtime/state").json()
    assert state["kv_cache"]["occupancy"] == 0.0 and state["kv_cache"]["free_pages"] == 64
    assert state["prefix_cache"]["hit_rate"] == 0.0 and state["prefix_cache"]["miss_pages"] == 0


def test_the_step_profile_is_read_and_reset_over_http(tok):
    app, engine, _ = make_app(tok)
    body = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4}
    with TestClient(app) as client:
        for _ in range(3):
            assert client.post("/v1/chat/completions", json=body).status_code == 200
        profile = client.get("/v1/runtime/profile").json()
        assert profile["object"] == "runtime.profile" and profile["steps"] == 3 and "records" not in profile
        assert profile["totals"]["decode_ms"] == 6.0 and profile["idle_ms"] == 24.0
        with_records = client.get("/v1/runtime/profile", params={"records": "true"}).json()
        assert [r["start_ms"] for r in with_records["records"]] == [0.0, 10.0, 20.0]
        assert client.delete("/v1/runtime/profile").json() == {"object": "runtime.profile", "reset": True}
        assert client.get("/v1/runtime/profile").json()["steps"] == 0

