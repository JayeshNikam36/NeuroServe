"""FastAPI application: OpenAI-compatible endpoints backed by NeuroServe."""

from __future__ import annotations

import contextvars
import functools
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import CudaError, KvCacheExhausted, __version__
from .config import ServerConfig
from .engine_service import ContextLengthExceeded, EngineService, GenerationResult
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
    ResponseMessage,
    Usage,
)

logger = logging.getLogger("neuroserve.server")


class ApiError(Exception):
    """An error returned to the client in OpenAI's error format."""

    def __init__(self, status_code: int, message: str, *, type_: str = "invalid_request_error",
                 param: str | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.type = type_
        self.param = param
        self.code = code


def _error(status_code: int, message: str, type_: str, param: str | None = None,
           code: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": type_, "param": param, "code": code}},
    )


def _check_request(config: ServerConfig, *, model: str, stream: bool, n: int, temperature: float | None,
                   max_tokens: int | None) -> int:
    """Rejects options NeuroServe does not support yet; returns the max_tokens to use."""
    if model != config.model.name:
        raise ApiError(404, f"The model '{model}' does not exist; this server serves '{config.model.name}'",
                       param="model", code="model_not_found")
    if stream:
        raise ApiError(400, "stream=true is not supported yet", param="stream", code="unsupported_parameter")
    if n != 1:
        raise ApiError(400, "only n=1 is supported", param="n", code="unsupported_parameter")
    if temperature not in (None, 0):
        raise ApiError(400, "only greedy decoding is supported yet: omit temperature or set it to 0",
                       param="temperature", code="unsupported_parameter")
    if max_tokens is not None and max_tokens > config.limits.max_tokens_limit:
        raise ApiError(400, f"max_tokens {max_tokens} exceeds the limit of {config.limits.max_tokens_limit}",
                       param="max_tokens")
    return max_tokens or config.limits.default_max_tokens


async def _run(request: Request, call: Callable[[], GenerationResult], param: str) -> GenerationResult:
    """Runs generation in a worker thread (the C++ engine releases the GIL) and maps errors."""
    failure = request.app.state.failure
    if failure is not None:
        raise ApiError(503, f"service unavailable after a GPU error ({failure}); restart the server",
                       type_="server_error", code="service_unavailable")

    context = contextvars.copy_context()  # carries the request id into the worker thread's log records
    try:
        return await run_in_threadpool(context.run, call)
    except ContextLengthExceeded as exc:
        raise ApiError(413, str(exc), param=param, code="context_length_exceeded") from exc
    except KvCacheExhausted as exc:
        raise ApiError(503, f"KV cache exhausted: {exc}", type_="server_error", code="kv_cache_exhausted") from exc
    except CudaError as exc:
        # A CUDA failure can leave the GPU context broken: stop serving instead of returning garbage.
        request.app.state.failure = str(exc)
        logger.error("GPU error; server marked unhealthy", extra={"error": str(exc)})
        raise ApiError(500, f"GPU error while generating (request id {request_id_var.get()})",
                       type_="server_error", code="gpu_error") from exc
    except ValueError as exc:  # invalid messages (ChatTemplateError), bad token ids, other bad input
        raise ApiError(400, str(exc), param=param) from exc


def _usage(result: GenerationResult) -> Usage:
    return Usage(prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                 total_tokens=result.prompt_tokens + result.completion_tokens)


def _info(result: GenerationResult, options: NeuroServeOptions) -> NeuroServeInfo:
    return NeuroServeInfo(
        e2e_ms=round(result.e2e_ms, 2),
        generated_tokens=result.generated_tokens,
        completion_token_ids=list(result.token_ids) if options.return_token_ids else None,
    )


def _record(request: Request, config: ServerConfig, prompt: Any, result: GenerationResult) -> None:
    """Adds usage numbers to the access log line; logs content only when explicitly enabled."""
    request.state.log_fields.update(
        prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
        generated_tokens=result.generated_tokens, finish_reason=result.finish_reason,
        e2e_ms=round(result.e2e_ms, 2),
    )
    if config.logging.log_prompts:
        logger.info("prompt and completion", extra={"prompt_logging": True, "prompt": prompt,
                                                    "completion": result.text})


def create_app(config: ServerConfig, service: EngineService | None = None) -> FastAPI:
    """Builds the app. Without a service, the real model is loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service if service is not None else await run_in_threadpool(EngineService.load, config)
        app.state.created = int(time.time())
        app.state.failure = None  # set after a GPU error; the server then refuses work
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
        return _error(exc.status_code, exc.message, exc.type, exc.param, exc.code)

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
        return {"status": "ok", "model": config.model.name, "gpu": svc.device_name, "kv_cache": svc.kv_stats()}

    @app.get("/v1/models")
    async def models(request: Request) -> dict:
        return {
            "object": "list",
            "data": [{"id": config.model.name, "object": "model",
                      "created": request.app.state.created, "owned_by": "neuroserve"}],
        }

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(body: ChatCompletionRequest, request: Request) -> ChatCompletionResponse:
        svc: EngineService = request.app.state.service
        max_tokens = _check_request(config, model=body.model, stream=body.stream, n=body.n,
                                    temperature=body.temperature,
                                    max_tokens=body.max_completion_tokens or body.max_tokens)
        messages = [m.model_dump() for m in body.messages]
        result = await _run(request, functools.partial(svc.chat, messages, max_tokens,
                                                       ignore_eos=body.neuroserve.ignore_eos), param="messages")
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
    async def completions(body: CompletionRequest, request: Request) -> CompletionResponse:
        svc: EngineService = request.app.state.service
        max_tokens = _check_request(config, model=body.model, stream=body.stream, n=body.n,
                                    temperature=body.temperature, max_tokens=body.max_tokens)
        result = await _run(request, functools.partial(svc.complete, body.prompt, max_tokens,
                                                       ignore_eos=body.neuroserve.ignore_eos), param="prompt")
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
