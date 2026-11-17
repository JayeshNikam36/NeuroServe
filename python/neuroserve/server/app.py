"""FastAPI application: OpenAI-compatible endpoints backed by NeuroServe."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import (CudaError, EngineStopped, KvCacheExhausted, QueueFull, QueueTimeout, RequestCancelled, __version__,
                 scheduling_policies)
from .config import ServerConfig
from .engine_service import ContextLengthExceeded, EngineService, GenerationResult, StreamingGeneration
from .logging_setup import request_id_var
from .middleware import RequestContextMiddleware
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    NeuroServeInfo,
    NeuroServeOptions,
    PolicyChange,
    ResponseMessage,
    StreamOptions,
    Usage,
)

T = TypeVar("T")

logger = logging.getLogger("neuroserve.server")

DISCONNECT_POLL_S = 0.1  # how often a non-streaming request checks whether its client is still there


class ApiError(Exception):
    """An error returned to the client in OpenAI's error format."""

    def __init__(self, status_code: int, message: str, *, type_: str = "invalid_request_error",
                 param: str | None = None, code: str | None = None, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.type = type_
        self.param = param
        self.code = code
        self.headers = headers  # e.g. Retry-After for 429 and 503


def _error(status_code: int, message: str, type_: str, param: str | None = None,
           code: str | None = None, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": type_, "param": param, "code": code}},
        headers=headers,
    )


RETRY_AFTER = {"Retry-After": "1"}  # overload errors: the client may retry after a second


def _check_request(config: ServerConfig, *, model: str, stream: bool, stream_options: StreamOptions | None,
                   n: int, temperature: float | None, max_tokens: int | None) -> int:
    """Rejects options NeuroServe does not support yet; returns the max_tokens to use."""
    if model != config.model.name:
        raise ApiError(404, f"The model '{model}' does not exist; this server serves '{config.model.name}'",
                       param="model", code="model_not_found")
    if stream_options is not None and not stream:
        raise ApiError(400, "stream_options is only allowed with stream=true", param="stream_options")
    if n != 1:
        raise ApiError(400, "only n=1 is supported", param="n", code="unsupported_parameter")
    if temperature not in (None, 0):
        raise ApiError(400, "only greedy decoding is supported yet: omit temperature or set it to 0",
                       param="temperature", code="unsupported_parameter")
    if max_tokens is not None and max_tokens > config.limits.max_tokens_limit:
        raise ApiError(400, f"max_tokens {max_tokens} exceeds the limit of {config.limits.max_tokens_limit}",
                       param="max_tokens")
    return max_tokens or config.limits.default_max_tokens


def _api_error(request: Request, exc: Exception, param: str) -> ApiError | None:
    """The client-facing error for an exception from the service or engine (None: a bug)."""
    if isinstance(exc, ContextLengthExceeded):
        return ApiError(413, str(exc), param=param, code="context_length_exceeded")
    if isinstance(exc, QueueFull):
        return ApiError(429, f"the server is busy: {exc}; retry later", type_="server_error", code="queue_full",
                        headers=RETRY_AFTER)
    if isinstance(exc, QueueTimeout):
        return ApiError(503, f"the server is busy: {exc}", type_="server_error", code="queue_timeout",
                        headers=RETRY_AFTER)
    if isinstance(exc, RequestCancelled):
        # 499 (nginx's "client closed request"): cancelled by a disconnect or DELETE /v1/requests/{id}.
        return ApiError(499, "the request was cancelled", code="request_cancelled")
    if isinstance(exc, EngineStopped):
        return ApiError(503, "the engine is shutting down", type_="server_error", code="service_unavailable")
    if isinstance(exc, KvCacheExhausted):
        return ApiError(503, f"KV cache exhausted: {exc}", type_="server_error", code="kv_cache_exhausted")
    if isinstance(exc, CudaError):
        # A CUDA failure can leave the GPU context broken: stop serving instead of returning garbage.
        request.app.state.failure = str(exc)
        logger.error("GPU error; server marked unhealthy", extra={"error": str(exc)})
        return ApiError(500, f"GPU error while generating (request id {request_id_var.get()})",
                        type_="server_error", code="gpu_error")
    if isinstance(exc, ValueError):  # invalid messages (ChatTemplateError), bad token ids, other bad input
        return ApiError(400, str(exc), param=param)
    return None


async def _run(request: Request, call: Callable[[], T], param: str) -> T:
    """Runs a service call in a worker thread (the C++ engine releases the GIL) and maps errors."""
    failure = request.app.state.failure
    if failure is not None:
        raise ApiError(503, f"service unavailable after a GPU error ({failure}); restart the server",
                       type_="server_error", code="service_unavailable")

    context = contextvars.copy_context()  # carries the request id into the worker thread's log records
    try:
        return await run_in_threadpool(context.run, call)
    except Exception as exc:
        error = _api_error(request, exc, param)
        if error is None:
            raise
        raise error from exc


def _track(request: Request, generation: StreamingGeneration) -> None:
    """Makes the request cancellable with DELETE /v1/requests/{its X-Request-ID}."""
    request.app.state.in_flight.setdefault(request.state.request_id, set()).add(generation.engine_request_id)


def _untrack(request: Request, generation: StreamingGeneration) -> None:
    in_flight: dict[str, set[str]] = request.app.state.in_flight
    ids = in_flight.get(request.state.request_id)
    if ids is not None:
        ids.discard(generation.engine_request_id)
        if not ids:
            del in_flight[request.state.request_id]


async def _finish(request: Request, generation: StreamingGeneration, param: str) -> GenerationResult:
    """Waits for a non-streaming generation; cancels it if the client disconnects first."""
    _track(request, generation)
    waiting = asyncio.ensure_future(_run(request, generation.result, param))
    try:
        while True:
            done, _ = await asyncio.wait({waiting}, timeout=DISCONNECT_POLL_S)
            if done:
                return waiting.result()
            if await request.is_disconnected():
                logger.info("client disconnected; cancelling the request",
                            extra={"engine_request_id": generation.engine_request_id})
                generation.cancel()
                return await waiting  # raises the 499; nobody is listening any more
    finally:
        _untrack(request, generation)


def _usage(result: GenerationResult) -> Usage:
    return Usage(prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                 total_tokens=result.prompt_tokens + result.completion_tokens)


def _info(result: GenerationResult, options: NeuroServeOptions) -> NeuroServeInfo:
    return NeuroServeInfo(
        e2e_ms=round(result.e2e_ms, 2),
        ttft_ms=round(result.ttft_ms, 2),
        queue_ms=round(result.queue_ms, 2),
        itl_ms=round(result.itl_ms, 2),
        max_itl_ms=round(result.max_itl_ms, 2),
        generated_tokens=result.generated_tokens,
        completion_token_ids=list(result.token_ids) if options.return_token_ids else None,
    )


def _record(request: Request, config: ServerConfig, prompt: Any, result: GenerationResult, *,
            streamed: bool = False) -> None:
    """Adds usage numbers to the access log line; logs content only when explicitly enabled."""
    fields = dict(
        prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
        generated_tokens=result.generated_tokens, finish_reason=result.finish_reason,
        e2e_ms=round(result.e2e_ms, 2), ttft_ms=round(result.ttft_ms, 2),
        queue_ms=round(result.queue_ms, 2), itl_ms=round(result.itl_ms, 2),
        engine_request_id=result.engine_request_id,
    )
    if streamed:
        # The access line was written when the stream started; this line closes it.
        logger.info("stream completed", extra={"path": request.url.path, **fields})
    else:
        request.state.log_fields.update(fields)
    if config.logging.log_prompts:
        logger.info("prompt and completion", extra={"prompt_logging": True, "prompt": prompt,
                                                    "completion": result.text})


def _sse(data: dict[str, Any] | str) -> str:
    """One server-sent event."""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"data: {payload}\n\n"


def _stream(request: Request, config: ServerConfig, generation: StreamingGeneration, *, chat: bool,
            options: NeuroServeOptions, stream_options: StreamOptions | None, prompt: Any) -> StreamingResponse:
    """Streams a generation as OpenAI chunks over server-sent events, ending with `data: [DONE]`.

    Chat: a first chunk with the role, one chunk per piece of text, then a chunk with the
    finish_reason. With stream_options.include_usage, a last chunk with empty choices and the usage.
    The last chunk before [DONE] carries the neuroserve block (timings, token ids on request).
    """
    request_id = request.state.request_id
    response_id = f"chatcmpl-{uuid.uuid4().hex}" if chat else f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    include_usage = stream_options is not None and stream_options.include_usage

    def chunk(choices: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
        return {"id": response_id, "object": "chat.completion.chunk" if chat else "text_completion",
                "created": created, "model": config.model.name, "system_fingerprint": None,
                "choices": choices, **extra}

    def choice(text: str | None = None, finish_reason: str | None = None, role: bool = False) -> dict[str, Any]:
        if not chat:
            return {"index": 0, "text": text or "", "logprobs": None, "finish_reason": finish_reason}
        delta: dict[str, Any] = {"role": "assistant"} if role else {}
        if text is not None:
            delta["content"] = text
        return {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}

    async def events() -> AsyncIterator[str]:
        token = request_id_var.set(request_id)  # the stream outlives the middleware's context
        finished = False
        try:
            if chat:
                yield _sse(chunk([choice("", role=True)]))
            # deltas() blocks on the engine: each next() runs in a worker thread (GIL released) and
            # returns at least every 0.1 s, so a disconnect cancels the stream promptly.
            async for text in iterate_in_threadpool(generation.deltas()):
                if text:  # "" means nothing new yet
                    yield _sse(chunk([choice(text)]))
            result = await run_in_threadpool(generation.result)
            info = _info(result, options).model_dump()
            if include_usage:
                yield _sse(chunk([choice(finish_reason=result.finish_reason)]))
                yield _sse(chunk([], usage=_usage(result).model_dump(), neuroserve=info))
            else:
                yield _sse(chunk([choice(finish_reason=result.finish_reason)], neuroserve=info))
            yield _sse("[DONE]")
            finished = True
            _record(request, config, prompt, result, streamed=True)
        except Exception as exc:
            # The 200 status is already sent: report the error in the stream, as OpenAI does.
            error = _api_error(request, exc, "messages" if chat else "prompt")
            if error is None:
                logger.exception("unhandled error while streaming %s", request.url.path)
                error = ApiError(500, f"internal server error (request id {request_id})",
                                 type_="server_error", code="internal_error")
            finished = True
            yield _sse({"error": {"message": error.message, "type": error.type, "param": error.param,
                                  "code": error.code}})
        finally:
            # Not finished here means the client went away mid-stream (the task was cancelled).
            if not finished and generation.cancel():
                logger.info("client disconnected; cancelling the request",
                            extra={"engine_request_id": generation.engine_request_id})
            _untrack(request, generation)
            request_id_var.reset(token)

    _track(request, generation)
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def create_app(config: ServerConfig, service: EngineService | None = None) -> FastAPI:
    """Builds the app. Without a service, the real model is loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service if service is not None else await run_in_threadpool(EngineService.load, config)
        app.state.created = int(time.time())
        app.state.failure = None  # set after a GPU error; the server then refuses work
        app.state.in_flight = {}  # X-Request-ID -> engine request ids, for DELETE /v1/requests/{id}
        logger.info("NeuroServe %s ready: model=%s gpu=%s", __version__, config.model.name,
                    app.state.service.device_name)
        try:
            yield
        finally:
            app.state.service = None  # releases the engine and its GPU memory

    app = FastAPI(title="NeuroServe", version=__version__, lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status_code, exc.message, exc.type, exc.param, exc.code, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        first = errors[0] if errors else {}
        loc = [str(p) for p in first.get("loc", ()) if p != "body"]
        param = ".".join(loc) or None
        message = first.get("msg", "invalid request")
        return _error(400, f"{param}: {message}" if param else message, "invalid_request_error", param)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, str(exc.detail), "invalid_request_error")

    @app.get("/health")
    async def health(request: Request):
        if request.app.state.failure is not None:
            return JSONResponse(status_code=503, content={
                "status": "unhealthy", "reason": request.app.state.failure, "model": config.model.name})
        svc: EngineService = request.app.state.service
        return {"status": "ok", "model": config.model.name, "gpu": svc.device_name, "kv_cache": svc.kv_stats(),
                "engine": svc.engine_stats(), "latency": svc.latency()}

    @app.get("/v1/models")
    async def models(request: Request) -> dict:
        return {
            "object": "list",
            "data": [{"id": config.model.name, "object": "model",
                      "created": request.app.state.created, "owned_by": "neuroserve"}],
        }

    @app.get("/v1/runtime/decisions")
    async def decisions(request: Request, since: int = Query(default=0, ge=0),
                        limit: int = Query(default=100, ge=1, le=1000)) -> dict:
        """The scheduler's explanations newer than `since`, oldest first (see docs/scheduling.md)."""
        svc: EngineService = request.app.state.service
        page = svc.decisions(since, limit)
        return {"object": "list", "data": page["data"], "next_since": page["next_since"],
                "dropped": page["dropped"]}

    @app.get("/v1/runtime/policy")
    async def get_policy(request: Request) -> dict:
        svc: EngineService = request.app.state.service
        return {"object": "runtime.policy", "policy": svc.scheduling_policy(), "available": list(scheduling_policies)}

    @app.put("/v1/runtime/policy")
    async def set_policy(body: PolicyChange, request: Request) -> dict:
        """Switches the scheduling policy from the next engine step on (A/B benchmarks)."""
        svc: EngineService = request.app.state.service
        try:
            previous = svc.set_scheduling_policy(body.policy)
        except ValueError as exc:
            raise ApiError(400, str(exc), param="policy", code="unknown_policy") from exc
        logger.info("scheduling policy changed", extra={"policy": body.policy, "previous_policy": previous})
        return {"object": "runtime.policy", "policy": body.policy, "previous": previous,
                "available": list(scheduling_policies)}

    @app.delete("/v1/requests/{request_id}")
    async def cancel_request(request_id: str, request: Request) -> dict:
        """Cancels the in-flight request(s) sent with this X-Request-ID."""
        svc: EngineService = request.app.state.service
        engine_ids = request.app.state.in_flight.get(request_id, set())
        cancelled = [engine_id for engine_id in sorted(engine_ids) if svc.cancel(engine_id)]
        if not cancelled:
            raise ApiError(404, f"no request with id '{request_id}' is in progress", param="request_id",
                           code="request_not_found")
        logger.info("request cancelled by DELETE", extra={"cancelled_request_id": request_id,
                                                         "engine_request_ids": cancelled})
        return {"id": request_id, "object": "request.cancellation", "cancelled": True}

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(body: ChatCompletionRequest,
                               request: Request) -> ChatCompletionResponse | StreamingResponse:
        svc: EngineService = request.app.state.service
        max_tokens = _check_request(config, model=body.model, stream=body.stream, stream_options=body.stream_options,
                                    n=body.n, temperature=body.temperature,
                                    max_tokens=body.max_completion_tokens or body.max_tokens)
        messages = [m.model_dump() for m in body.messages]
        if body.stream:
            # Bad requests fail here, with a normal HTTP error, before the stream starts.
            generation = await _run(request, functools.partial(svc.start_chat, messages, max_tokens,
                                                               ignore_eos=body.neuroserve.ignore_eos,
                                                               priority=body.neuroserve.priority), param="messages")
            return _stream(request, config, generation, chat=True, options=body.neuroserve,
                           stream_options=body.stream_options, prompt=messages)
        generation = await _run(request, functools.partial(svc.start_chat, messages, max_tokens,
                                                           ignore_eos=body.neuroserve.ignore_eos,
                                                           priority=body.neuroserve.priority), param="messages")
        result = await _finish(request, generation, param="messages")
        _record(request, config, messages, result)
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=config.model.name,
            choices=[Choice(index=0, message=ResponseMessage(content=result.text),
                            finish_reason=result.finish_reason)],
            usage=_usage(result),
            neuroserve=_info(result, body.neuroserve),
        )

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(body: CompletionRequest, request: Request) -> CompletionResponse | StreamingResponse:
        svc: EngineService = request.app.state.service
        max_tokens = _check_request(config, model=body.model, stream=body.stream, stream_options=body.stream_options,
                                    n=body.n, temperature=body.temperature, max_tokens=body.max_tokens)
        if body.stream:
            generation = await _run(request, functools.partial(svc.start_completion, body.prompt, max_tokens,
                                                               ignore_eos=body.neuroserve.ignore_eos,
                                                               priority=body.neuroserve.priority), param="prompt")
            return _stream(request, config, generation, chat=False, options=body.neuroserve,
                           stream_options=body.stream_options, prompt=body.prompt)
        generation = await _run(request, functools.partial(svc.start_completion, body.prompt, max_tokens,
                                                           ignore_eos=body.neuroserve.ignore_eos,
                                                           priority=body.neuroserve.priority), param="prompt")
        result = await _finish(request, generation, param="prompt")
        _record(request, config, body.prompt, result)
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=config.model.name,
            choices=[CompletionChoice(index=0, text=result.text, finish_reason=result.finish_reason)],
            usage=_usage(result),
            neuroserve=_info(result, body.neuroserve),
        )

    return app
