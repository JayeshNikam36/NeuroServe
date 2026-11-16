"""Per-request id, structured access log, and last-resort error handling."""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .logging_setup import request_id_var

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

access_logger = logging.getLogger("neuroserve.access")
error_logger = logging.getLogger("neuroserve.server")


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex}"


class RequestContextMiddleware:
    """Assigns a request id, logs one access line per request, and turns any unhandled
    exception into an OpenAI-format 500 (the traceback goes to the log, never to the client).

    A plain ASGI middleware: the app runs in the same task and sees the client's own receive
    channel, so an endpoint (or a stream) notices when the client disconnects. The access line
    is written when the response starts; for a stream that is before its body is sent.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = incoming if incoming and _VALID_REQUEST_ID.match(incoming) else new_request_id()
        token = request_id_var.set(request_id)
        state = scope.setdefault("state", {})  # what request.state reads and writes
        state["request_id"] = request_id
        state["log_fields"] = {}  # routes add usage numbers here for the access log
        method, path = scope["method"], scope["path"]
        start = time.perf_counter()
        response_started = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
                status = message["status"]
                access_logger.info(
                    "%s %s %d", method, path, status,
                    extra={"method": method, "path": path, "status": status,
                           "duration_ms": round((time.perf_counter() - start) * 1000.0, 2),
                           **state["log_fields"]},
                )
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            error_logger.exception("unhandled error while serving %s %s", method, path)
            if not response_started:  # otherwise the client already has a status; the connection just ends
                response = JSONResponse(status_code=500, content={"error": {
                    "message": f"internal server error (request id {request_id})",
                    "type": "server_error", "param": None, "code": "internal_error"}})
                await response(scope, receive, send_with_request_id)
        finally:
            request_id_var.reset(token)
