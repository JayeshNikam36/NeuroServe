"""Structured logging, request ids, prompt privacy, and last-resort error handling."""

import io
import json
import logging

import pytest
from fastapi.testclient import TestClient

import neuroserve
from neuroserve.server.app import create_app
from neuroserve.server.engine_service import EngineService
from neuroserve.server.logging_setup import configure_logging

from test_api import TOKENIZER_DIR, FakeEngine, chat_body, make_config, tok  # noqa: F401  (tok is a fixture)

pytestmark = pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer.json").exists(),
                                reason=f"tokenizer not found in {TOKENIZER_DIR}")

SECRET = "SECRET-PROMPT-7731"


class LoggingEngine(FakeEngine):
    """A fake engine that logs from inside generate(), i.e. from the worker thread."""

    def generate(self, prompt_ids, max_new_tokens):
        logging.getLogger("test.engine").info("generating inside the worker thread")
        return super().generate(prompt_ids, max_new_tokens)


@pytest.fixture
def log_buffer():
    buf = io.StringIO()
    configure_logging("INFO", "json", stream=buf)
    yield buf
    configure_logging("WARNING", "text")


def records(buf: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]


def build_app(tok, *, error=None, log_prompts=False, engine_cls=FakeEngine):
    eot = tok.tokenizer.token_to_id("<|eot_id|>")
    reply_ids = tok.encode("Paris.", add_special_tokens=False) + [eot]
    filler = tok.encode(" filler", add_special_tokens=False)[0]
    config = make_config()
    if log_prompts:
        config = config.model_copy(update={"logging": config.logging.model_copy(update={"log_prompts": True})})
    engine = engine_cls(reply_ids, filler, error)
    return create_app(config, EngineService(config, engine, tok, device_name="fake-gpu"))


def test_every_response_has_a_request_id(tok):
    with TestClient(build_app(tok)) as client:
        for resp in (client.get("/health"), client.post("/v1/chat/completions", json=chat_body()),
                     client.get("/no/such/path")):
            assert resp.headers["X-Request-ID"].startswith("req_")


def test_valid_client_request_id_is_kept_and_unsafe_one_replaced(tok):
    with TestClient(build_app(tok)) as client:
        kept = client.get("/health", headers={"X-Request-ID": "client-abc.123"})
        replaced = client.get("/health", headers={"X-Request-ID": 'bad id\n{"forged": true}'})
    assert kept.headers["X-Request-ID"] == "client-abc.123"
    assert replaced.headers["X-Request-ID"].startswith("req_")


def test_access_log_is_json_with_request_id_and_usage(tok, log_buffer):
    with TestClient(build_app(tok)) as client:
        resp = client.post("/v1/chat/completions", json=chat_body())
    rid = resp.headers["X-Request-ID"]
    access = [r for r in records(log_buffer) if r["logger"] == "neuroserve.access"]
    line = next(r for r in access if r["path"] == "/v1/chat/completions")
    assert line["request_id"] == rid and line["status"] == 200 and line["method"] == "POST"
    assert line["duration_ms"] >= 0 and line["prompt_tokens"] > 0
    assert line["completion_tokens"] == resp.json()["usage"]["completion_tokens"]
    assert line["finish_reason"] == "stop"


def test_request_id_reaches_logs_written_in_the_worker_thread(tok, log_buffer):
    with TestClient(build_app(tok, engine_cls=LoggingEngine)) as client:
        rid = client.post("/v1/chat/completions", json=chat_body()).headers["X-Request-ID"]
    inner = [r for r in records(log_buffer) if r["logger"] == "test.engine"]
    assert inner and inner[0]["request_id"] == rid


def test_prompts_are_not_logged_by_default(tok, log_buffer):
    body = chat_body(messages=[{"role": "user", "content": SECRET}])
    with TestClient(build_app(tok)) as client:
        assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert SECRET not in log_buffer.getvalue()


def test_prompt_logging_is_opt_in(tok, log_buffer):
    body = chat_body(messages=[{"role": "user", "content": SECRET}])
    with TestClient(build_app(tok, log_prompts=True)) as client:
        client.post("/v1/chat/completions", json=body)
    logged = [r for r in records(log_buffer) if r.get("prompt_logging")]
    assert logged and SECRET in json.dumps(logged[0]["prompt"])


def test_unexpected_error_is_a_clean_500(tok, log_buffer):
    with TestClient(build_app(tok, error=RuntimeError("boom in engine"))) as client:
        resp = client.post("/v1/chat/completions", json=chat_body())
    rid = resp.headers["X-Request-ID"]
    assert resp.status_code == 500
    error = resp.json()["error"]
    assert error["type"] == "server_error" and error["code"] == "internal_error" and rid in error["message"]
    assert "Traceback" not in resp.text and "boom" not in resp.text  # no internals leak to the client
    logged = [r for r in records(log_buffer) if r["level"] == "ERROR" and r["request_id"] == rid]
    assert logged and "RuntimeError: boom in engine" in logged[0]["exc"]  # full detail in the log


def test_gpu_error_marks_server_unhealthy(tok, log_buffer):
    with TestClient(build_app(tok, error=neuroserve.CudaError("an illegal memory access was encountered"))) as client:
        first = client.post("/v1/chat/completions", json=chat_body())
        health = client.get("/health")
        second = client.post("/v1/chat/completions", json=chat_body())
    assert first.status_code == 500 and first.json()["error"]["code"] == "gpu_error"
    assert health.status_code == 503 and health.json()["status"] == "unhealthy"
    assert second.status_code == 503 and second.json()["error"]["code"] == "service_unavailable"
    assert any(r["msg"] == "GPU error; server marked unhealthy" for r in records(log_buffer))
