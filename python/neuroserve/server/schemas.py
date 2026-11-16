"""OpenAI-compatible request and response models (the subset NeuroServe supports)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NeuroServeOptions(BaseModel):
    """NeuroServe request extension. OpenAI clients that do not send it get the defaults."""

    model_config = ConfigDict(extra="forbid")

    return_token_ids: bool = False  # include the completion token ids in the response
    ignore_eos: bool = False        # generate exactly max_tokens; do not stop at stop tokens
    priority: int = Field(default=1, ge=0, le=2)  # 0 low, 1 normal, 2 high (used by the "priority" policy)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    include_usage: bool = False  # send a last chunk with usage (and the neuroserve block)


class _GenerationRequest(BaseModel):
    # OpenAI clients may send fields NeuroServe does not use; ignore them.
    model_config = ConfigDict(extra="ignore")

    model: str
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    user: str | None = None
    neuroserve: NeuroServeOptions = Field(default_factory=NeuroServeOptions)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(_GenerationRequest):
    messages: list[ChatMessage] = Field(min_length=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)  # newer OpenAI name


class CompletionRequest(_GenerationRequest):
    prompt: str | list[int]  # text, or token ids used verbatim

    @field_validator("prompt")
    @classmethod
    def _not_empty(cls, value: str | list[int]) -> str | list[int]:
        if len(value) == 0:
            raise ValueError("prompt must not be empty")
        return value


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class NeuroServeInfo(BaseModel):
    """NeuroServe response extension: timing, engine work and (on request) token ids."""

    e2e_ms: float
    ttft_ms: float         # time to first token, measured in the engine
    queue_ms: float        # time spent waiting for admission
    itl_ms: float          # mean inter-token latency (time between consecutive tokens)
    max_itl_ms: float      # the longest gap between two tokens
    generated_tokens: int  # tokens the engine produced (stops on the stop token: equals completion_tokens)
    completion_token_ids: list[int] | None = None  # returned tokens, including the stop token if hit


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: Literal["stop", "length"]
    logprobs: None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    system_fingerprint: str | None = None
    neuroserve: NeuroServeInfo


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Literal["stop", "length"]
    logprobs: None = None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage
    system_fingerprint: str | None = None
    neuroserve: NeuroServeInfo
