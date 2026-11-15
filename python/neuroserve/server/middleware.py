"""Per-request id, structured access log, and last-resort error handling."""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .logging_setup import request_id_var

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

access_logger = logging.getLogger("neuroserve.access")
error_logger = logging.getLogger("neuroserve.server")


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex}"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, logs one access line per request, and turns any unhandled
    exception into an OpenAI-format 500 (the traceback goes to the log, never to the client)."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = incoming if incoming and _VALID_REQUEST_ID.match(incoming) else new_request_id()
        token = request_id_var.set(request_id)
        request.state.request_id = request_id
        request.state.log_fields = {}  # routes add usage numbers here for the access log
        start = time.perf_counter()
        try:
            try:
                response = await call_next(request)
            except Exception:
                error_logger.exception("unhandled error while serving %s %s", request.method, request.url.path)
                response = JSONResponse(status_code=500, content={"error": {
                    "message": f"internal server error (request id {request_id})",
                    "type": "server_error", "param": None, "code": "internal_error"}})
            response.headers[REQUEST_ID_HEADER] = request_id
            access_logger.info(
                "%s %s %d", request.method, request.url.path, response.status_code,
                extra={"method": request.method, "path": request.url.path, "status": response.status_code,
                       "duration_ms": round((time.perf_counter() - start) * 1000.0, 2),
                       **request.state.log_fields},
            )
            return response
        finally:
            request_id_var.reset(token)
