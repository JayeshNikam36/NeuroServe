"""End-to-end correctness through the HTTP API: responses must match Hugging Face token for token.

1. Base model: the 8 reference prompts sent as token ids to /v1/completions (ignore_eos).
2. Instruct model: the 5 chat cases sent as messages to /v1/chat/completions, with the template
   date pinned so the server builds exactly the prompt Hugging Face saw.
"""

import gc
import json
import os
from array import array
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neuroserve.server.app import create_app
from neuroserve.server.config import ServerConfig

BASE_DIR = Path(os.environ.get("NS_MODEL_DIR", "/models/llama-3.2-1b-fp16"))
INSTRUCT_DIR = Path(os.environ.get("NS_INSTRUCT_MODEL_DIR", "/models/llama-3.2-1b-instruct-fp16"))
TOKENIZER_DIR = Path(os.environ.get("NS_TOKENIZER_DIR", "/models/llama-3.2-1b-instruct"))
REF_DIR = Path(os.environ.get("NS_REF_DIR", "/models/llama-3.2-1b-ref"))

needs_base = pytest.mark.skipif(
    not ((BASE_DIR / "weights.bin").exists() and (REF_DIR / "prompt_00_ids.i32").exists()),
    reason="base weights or reference prompts not found")
needs_instruct = pytest.mark.skipif(
    not ((INSTRUCT_DIR / "weights.bin").exists() and (REF_DIR / "instruct_greedy.json").exists()),
    reason="instruct weights or reference not found")


def read_i32(path: Path) -> list[int]:
    values = array("i")
    values.frombytes(path.read_bytes())
    return values.tolist()


def make_config(name: str, weights_dir: Path, date_string: str | None) -> ServerConfig:
    return ServerConfig.model_validate({
        "model": {"name": name, "weights_dir": str(weights_dir), "tokenizer_dir": str(TOKENIZER_DIR),
                  "chat_date_string": date_string},
        "engine": {"max_prompt_tokens": 512, "num_pages": 512},
        "limits": {"default_max_tokens": 32, "max_tokens_limit": 64},
    })


@contextmanager
def serve(config: ServerConfig):
    with TestClient(create_app(config)) as client:
        yield client
    gc.collect()  # release the engine before the next model is loaded


@needs_base
def test_completions_with_reference_token_ids_match_hugging_face():
    with serve(make_config("llama-3.2-1b", BASE_DIR, None)) as client:
        for i in range(8):
            prompt = read_i32(REF_DIR / f"prompt_{i:02d}_ids.i32")
            expected = read_i32(REF_DIR / f"prompt_{i:02d}_greedy.i32")
            resp = client.post("/v1/completions", json={
                "model": "llama-3.2-1b", "prompt": prompt, "max_tokens": len(expected),
                "neuroserve": {"return_token_ids": True, "ignore_eos": True},
            })
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["neuroserve"]["completion_token_ids"] == expected, f"prompt {i} ({len(prompt)} tokens)"
            assert body["usage"] == {"prompt_tokens": len(prompt), "completion_tokens": len(expected),
                                     "total_tokens": len(prompt) + len(expected)}
        kv = client.get("/health").json()["kv_cache"]
        assert kv["free_pages"] == kv["total_pages"], "KV pages leaked"


@needs_instruct
def test_chat_completions_match_hugging_face():
    chat_ref = json.loads((REF_DIR / "chat_reference.json").read_text(encoding="utf-8"))
    greedy = json.loads((REF_DIR / "instruct_greedy.json").read_text(encoding="utf-8"))
    messages_by_name = {case["name"]: case["messages"] for case in chat_ref["chat"]}

    config = make_config("llama-3.2-1b-instruct", INSTRUCT_DIR, chat_ref["date_string"])
    with serve(config) as client:
        tok = client.app.state.service.tokenizer
        for case in greedy["cases"]:
            expected = case["greedy_ids"]
            resp = client.post("/v1/chat/completions", json={
                "model": "llama-3.2-1b-instruct",
                "messages": messages_by_name[case["name"]],
                "max_tokens": greedy["max_new_tokens"],
                "neuroserve": {"return_token_ids": True},
            })
            assert resp.status_code == 200, resp.text
            body = resp.json()
            name = case["name"]
            # Same prompt as Hugging Face (template, pinned date, special tokens)...
            assert body["usage"]["prompt_tokens"] == len(case["prompt_ids"]), f"{name}: prompt differs"
            # ...same tokens out, including the stop token...
            assert body["neuroserve"]["completion_token_ids"] == expected, f"{name}: tokens differ"
            # ...same finish reason and text.
            stopped = expected[-1] in tok.stop_token_ids
            choice = body["choices"][0]
            assert choice["finish_reason"] == ("stop" if stopped else "length"), name
            assert choice["message"]["content"] == tok.decode(expected[:-1] if stopped else expected), name
