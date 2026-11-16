"""Server configuration, loaded from YAML and validated at startup."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PAGE_SIZE_TOKENS = 16  # fixed by the engine's fast decode kernel


class _Strict(BaseModel):
    # Unknown keys are errors, so a typo in the YAML fails loudly instead of being ignored.
    model_config = ConfigDict(extra="forbid")


class ModelSection(_Strict):
    name: str
    weights_dir: Path
    tokenizer_dir: Path
    # Pins the chat template's "Today Date" line (e.g. "26 Jul 2024") so prompts are reproducible.
    # None: use today's date, like Hugging Face.
    chat_date_string: str | None = None


class EngineSection(_Strict):
    max_prompt_tokens: int = Field(default=512, ge=1)
    num_pages: int = Field(default=2048, ge=1)
    max_batch: int = Field(default=8, ge=1)
    max_batch_tokens: int = Field(default=512, ge=1)  # tokens one engine step may process (prefill + decode)
    max_pages_per_sequence: int = Field(default=64, ge=1)
    device_index: int = Field(default=0, ge=0)
    scheduling_policy: Literal["fcfs"] = "fcfs"  # which queued requests are admitted next


class LimitsSection(_Strict):
    default_max_tokens: int = Field(default=128, ge=1)
    max_tokens_limit: int = Field(default=512, ge=1)


class LoggingSection(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"
    # Prompts and completions can contain personal data: never logged unless explicitly enabled.
    log_prompts: bool = False


class ServerSection(_Strict):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class ServerConfig(_Strict):
    model: ModelSection
    engine: EngineSection = Field(default_factory=EngineSection)
    limits: LimitsSection = Field(default_factory=LimitsSection)
    logging: LoggingSection = Field(default_factory=LoggingSection)
    server: ServerSection = Field(default_factory=ServerSection)

    @model_validator(mode="after")
    def _check_limits(self) -> "ServerConfig":
        per_sequence = PAGE_SIZE_TOKENS * self.engine.max_pages_per_sequence
        if self.engine.max_prompt_tokens + self.limits.max_tokens_limit > per_sequence:
            raise ValueError(
                f"max_prompt_tokens ({self.engine.max_prompt_tokens}) + max_tokens_limit "
                f"({self.limits.max_tokens_limit}) exceeds the per-sequence capacity of {per_sequence} tokens"
            )
        if self.engine.max_prompt_tokens > self.engine.max_batch_tokens:
            raise ValueError(
                f"max_prompt_tokens ({self.engine.max_prompt_tokens}) exceeds max_batch_tokens "
                f"({self.engine.max_batch_tokens}): the longest prompt's prefill could never fit in a step"
            )
        if self.limits.default_max_tokens > self.limits.max_tokens_limit:
            raise ValueError("default_max_tokens must not exceed max_tokens_limit")
        return self


def load_config(path: str | Path) -> ServerConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ServerConfig.model_validate(data)
