"""API tests: OpenAI format, validation, errors (fake engine) and one real end-to-end run."""

import os
from pathlib import Path

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


class FakeEngine:
    """Returns a fixed reply, padded to exactly max_new_tokens like the real engine."""

    def __init__(self, reply_ids: list[int], filler: int, error: Exception | None = None) -> None:
        self.reply_ids = reply_ids
        self.filler = filler
        self.error = error
        self.calls: list[tuple[list[int], int]] = []

    def generate(self, prompt_ids, max_new_tokens):
        if self.error is not None:
            raise self.error
        self.calls.append((list(prompt_ids), max_new_tokens))
        out = list(self.reply_ids[:max_new_tokens])
        return out + [self.filler] * (max_new_tokens - len(out))

    def kv_stats(self):
        return {"page_size_tokens": 16, "total_pages": 64, "free_pages": 64, "max_pages_per_sequence": 64}


@pytest.fixture(scope="module")
def tok() -> ChatTokenizer:
    return ChatTokenizer(TOKENIZER_DIR)


def make_app(tok, *, stop=True, error=None):
    eot = tok.tokenizer.token_to_id("<|eot_id|>")
    reply_ids = tok.encode(REPLY, add_special_tokens=False) + ([eot] if stop else [])
    filler = tok.encode(" filler", add_special_tokens=False)[0]
    engine = FakeEngine(reply_ids, filler, error)
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
    ({"stream": True}, 400, "stream"),
    ({"temperature": 0.7}, 400, "temperature"),
    ({"n": 2}, 400, "n"),
    ({"max_tokens": 1000}, 400, "max_tokens"),
    ({"max_tokens": 0}, 400, "max_tokens"),
    ({"messages": []}, 400, "messages"),
    ({"messages": [{"role": "robot", "content": "hi"}]}, 400, "messages.0.role"),
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
