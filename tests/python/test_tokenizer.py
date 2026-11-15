"""Tokenizer tests: Hugging Face parity and streaming-safe decoding."""

import json
import os
from pathlib import Path

import pytest

from neuroserve.tokenizer import ChatTemplateError, ChatTokenizer

TOKENIZER_DIR = Path(os.environ.get("NS_TOKENIZER_DIR", "/models/llama-3.2-1b-instruct"))
CHAT_REF = Path(os.environ.get("NS_CHAT_REF", "/models/llama-3.2-1b-ref/chat_reference.json"))

pytestmark = pytest.mark.skipif(
    not ((TOKENIZER_DIR / "tokenizer.json").exists() and CHAT_REF.exists()),
    reason=f"tokenizer or reference not found ({TOKENIZER_DIR}, {CHAT_REF})",
)

STREAM_TEXTS = [
    "The capital of France is Paris.",
    "naïve café — 東京タワー 🎉🚀",
    "def f(x):\n    return x ** 2\n",
    "Mixed scripts: 你好, мир, عالم, 👩‍💻",
]


@pytest.fixture(scope="module")
def tok() -> ChatTokenizer:
    return ChatTokenizer(TOKENIZER_DIR)


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads(CHAT_REF.read_text(encoding="utf-8"))


def test_chat_template_matches_hugging_face(tok, ref):
    assert len(ref["chat"]) == 6
    for case in ref["chat"]:
        kwargs = {"add_generation_prompt": case["add_generation_prompt"], "date_string": ref["date_string"]}
        assert tok.render_chat(case["messages"], **kwargs) == case["text"], f"{case['name']}: text differs"
        assert tok.encode_chat(case["messages"], **kwargs) == case["ids"], f"{case['name']}: ids differ"


def test_plain_encoding_matches_hugging_face(tok, ref):
    for case in ref["plain"]:
        assert tok.encode(case["text"]) == case["ids"], case["text"]


def test_chat_prompt_has_exactly_one_bos(tok, ref):
    ids = tok.encode_chat([{"role": "user", "content": "hi"}], date_string=ref["date_string"])
    bos = tok.tokenizer.token_to_id("<|begin_of_text|>")
    assert ids.count(bos) == 1 and ids[0] == bos


def test_stop_tokens_include_end_of_turn(tok):
    assert tok.tokenizer.token_to_id("<|eot_id|>") in tok.stop_token_ids
    assert tok.tokenizer.token_to_id("<|end_of_text|>") in tok.stop_token_ids


@pytest.mark.parametrize("text", STREAM_TEXTS)
def test_streaming_decode_equals_full_decode(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    detok = tok.detokenizer()
    pieces = [detok.push(i) for i in ids] + [detok.flush()]
    assert "".join(pieces) == tok.decode(ids)
    assert all("\ufffd" not in p for p in pieces), f"half a character was emitted: {pieces!r}"


def test_streaming_decode_skips_special_tokens(tok):
    eot = tok.tokenizer.token_to_id("<|eot_id|>")
    ids = tok.encode("Hello", add_special_tokens=False) + [eot]
    detok = tok.detokenizer()
    assert "".join(detok.push(i) for i in ids) + detok.flush() == "Hello"


def test_invalid_messages_are_rejected(tok):
    with pytest.raises(ChatTemplateError):
        tok.encode_chat([])
    with pytest.raises(ChatTemplateError):
        tok.encode_chat([{"role": "robot", "content": "hi"}])
    with pytest.raises(ChatTemplateError):
        tok.encode_chat([{"role": "user", "content": 42}])
    assert issubclass(ChatTemplateError, ValueError)  # the API maps ValueError to HTTP 400
