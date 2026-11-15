"""FastAPI application: OpenAI-compatible endpoints backed by NeuroServe."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import KvCacheExhausted, __version__
from .config import ServerConfig
from .engine_service import ContextLengthExceeded, EngineService
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    NeuroServeInfo,
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


def _check_supported(body: ChatCompletionRequest, config: ServerConfig) -> None:
    if body.model != config.model.name:
        raise ApiError(404, f"The model '{body.model}' does not exist; this server serves '{config.model.name}'",
                       param="model", code="model_not_found")
    if body.stream:
        raise ApiError(400, "stream=true is not supported yet", param="stream", code="unsupported_parameter")
    if body.n != 1:
        raise ApiError(400, "only n=1 is supported", param="n", code="unsupported_parameter")
    if body.temperature not in (None, 0):
        raise ApiError(400, "only greedy decoding is supported yet: omit temperature or set it to 0",
                       param="temperature", code="unsupported_parameter")
    requested = body.max_completion_tokens or body.max_tokens
    if requested is not None and requested > config.limits.max_tokens_limit:
        raise ApiError(400, f"max_tokens {requested} exceeds the limit of {config.limits.max_tokens_limit}",
                       param="max_tokens")


def create_app(config: ServerConfig, service: EngineService | None = None) -> FastAPI:
    """Builds the app. Without a service, the real model is loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service if service is not None else await run_in_threadpool(EngineService.load, config)
        app.state.created = int(time.time())
        logger.info("NeuroServe %s ready: model=%s gpu=%s", __version__, config.model.name,
                    app.state.service.device_name)
        try:
            yield
        finally:
            app.state.service = None  # releases the engine and its GPU memory

    app = FastAPI(title="NeuroServe", version=__version__, lifespan=lifespan)

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
    async def health(request: Request) -> dict:
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
        _check_supported(body, config)
        max_tokens = body.max_completion_tokens or body.max_tokens or config.limits.default_max_tokens
        messages = [m.model_dump() for m in body.messages]

        try:
            # Runs in a worker thread; the C++ engine releases the GIL, so /health stays responsive.
            result = await run_in_threadpool(svc.chat, messages, max_tokens)
        except ContextLengthExceeded as exc:
            raise ApiError(413, str(exc), param="messages", code="context_length_exceeded") from exc
        except KvCacheExhausted as exc:
            raise ApiError(503, f"KV cache exhausted: {exc}", type_="server_error",
                           code="kv_cache_exhausted") from exc
        except ValueError as exc:  # invalid messages (ChatTemplateError) and other bad input
            raise ApiError(400, str(exc), param="messages") from exc

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=config.model.name,
            choices=[Choice(index=0, message=ResponseMessage(content=result.text),
                            finish_reason=result.finish_reason)],
            usage=Usage(prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                        total_tokens=result.prompt_tokens + result.completion_tokens),
            neuroserve=NeuroServeInfo(e2e_ms=round(result.e2e_ms, 2), generated_tokens=result.generated_tokens),
        )

    return app
