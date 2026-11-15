"""Tokenization for NeuroServe.

- ChatTokenizer: renders the model's own chat template the way Hugging Face does,
  then encodes it; plain-text encode/decode; stop-token ids.
- IncrementalDetokenizer: streaming decode that never emits half of a multi-byte character.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

VALID_ROLES = frozenset({"system", "user", "assistant", "tool", "ipython"})
REPLACEMENT_CHAR = "\ufffd"  # what a decoder produces for an incomplete UTF-8 sequence


class ChatTemplateError(ValueError):
    """Raised by a chat template (via raise_exception) or for invalid messages."""


def _raise_exception(message: str) -> None:
    raise ChatTemplateError(message)


def _tojson(value: Any, ensure_ascii: bool = False, indent: int | None = None,
            separators: tuple[str, str] | None = None, sort_keys: bool = False) -> str:
    # Same behaviour as Hugging Face's tojson filter (Jinja's built-in one HTML-escapes).
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _token_text(value: Any) -> str | None:
    """tokenizer_config.json stores special tokens as a string or as {"content": ...}."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return value.get("content")
    raise TypeError(f"unexpected special-token entry: {value!r}")


def _validate_messages(messages: Sequence[Mapping[str, Any]]) -> None:
    if not messages:
        raise ChatTemplateError("messages must not be empty")
    for i, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ChatTemplateError(f"message {i} must be an object with 'role' and 'content'")
        role = message.get("role")
        if role not in VALID_ROLES:
            raise ChatTemplateError(f"message {i}: unsupported role {role!r}")
        if not isinstance(message.get("content"), str):
            raise ChatTemplateError(f"message {i}: 'content' must be a string")


class IncrementalDetokenizer:
    """Turns a stream of token ids into text deltas without ever splitting a character.

    Keeps two offsets into the generated ids (the same approach as vLLM / TGI):
    text is released only when decoding the window no longer ends in U+FFFD.
    """

    def __init__(self, tokenizer: Tokenizer, *, skip_special_tokens: bool = True) -> None:
        self._tokenizer = tokenizer
        self._skip = skip_special_tokens
        self._ids: list[int] = []
        self._prefix_offset = 0
        self._read_offset = 0

    def _decode(self, ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=self._skip)

    def push(self, token_id: int) -> str:
        """Adds one token; returns the newly completed text (possibly empty)."""
        self._ids.append(token_id)
        prefix_text = self._decode(self._ids[self._prefix_offset:self._read_offset])
        full_text = self._decode(self._ids[self._prefix_offset:])
        if len(full_text) > len(prefix_text) and not full_text.endswith(REPLACEMENT_CHAR):
            delta = full_text[len(prefix_text):]
            self._prefix_offset = self._read_offset
            self._read_offset = len(self._ids)
            return delta
        return ""

    def flush(self) -> str:
        """Returns any text still held back (call once the stream has ended)."""
        prefix_text = self._decode(self._ids[self._prefix_offset:self._read_offset])
        full_text = self._decode(self._ids[self._prefix_offset:])
        self._prefix_offset = self._read_offset = len(self._ids)
        return full_text[len(prefix_text):]


class ChatTokenizer:
    """Hugging Face-compatible tokenizer and chat-template renderer for one model."""

    def __init__(self, tokenizer_dir: str | Path) -> None:
        directory = Path(tokenizer_dir)
        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))

        config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
        template = config.get("chat_template")
        if isinstance(template, list):  # some repositories store several named templates
            template = next((t["template"] for t in template if t.get("name") == "default"), None)
        if not template:
            raise ValueError(f"{directory / 'tokenizer_config.json'} has no chat_template")

        # Same Jinja settings as transformers' apply_chat_template.
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self._template = env.from_string(template)

        self.bos_token = _token_text(config.get("bos_token"))
        self.eos_token = _token_text(config.get("eos_token"))
        self.stop_token_ids = self._load_stop_token_ids(directory)

    def _load_stop_token_ids(self, directory: Path) -> frozenset[int]:
        ids: set[int] = set()
        if self.eos_token:
            eos_id = self._tokenizer.token_to_id(self.eos_token)
            if eos_id is not None:
                ids.add(eos_id)
        generation_config = directory / "generation_config.json"
        if generation_config.exists():
            eos = json.loads(generation_config.read_text(encoding="utf-8")).get("eos_token_id")
            if isinstance(eos, list):
                ids.update(eos)
            elif eos is not None:
                ids.add(eos)
        return frozenset(ids)

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def render_chat(self, messages: Sequence[Mapping[str, Any]], *, add_generation_prompt: bool = True,
                    date_string: str | None = None) -> str:
        """Formats messages with the model's chat template (date_string pins 'Today Date')."""
        _validate_messages(messages)
        extra: dict[str, Any] = {}
        if date_string is not None:
            extra["date_string"] = date_string
        return self._template.render(
            messages=[dict(m) for m in messages],
            tools=None,
            add_generation_prompt=add_generation_prompt,
            bos_token=self.bos_token or "",
            eos_token=self.eos_token or "",
            **extra,
        )

    def encode_chat(self, messages: Sequence[Mapping[str, Any]], *, add_generation_prompt: bool = True,
                    date_string: str | None = None) -> list[int]:
        """Token ids for a chat, identical to transformers' apply_chat_template(tokenize=True)."""
        text = self.render_chat(messages, add_generation_prompt=add_generation_prompt, date_string=date_string)
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Plain-text encoding; add_special_tokens adds <|begin_of_text|>."""
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def detokenizer(self, *, skip_special_tokens: bool = True) -> IncrementalDetokenizer:
        """A new streaming decoder for one response."""
        return IncrementalDetokenizer(self._tokenizer, skip_special_tokens=skip_special_tokens)
