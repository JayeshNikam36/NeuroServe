"""Structured logging: one JSON object per line, with the current request id on every record."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import TextIO

# Set per HTTP request by the middleware; read by every log record (also in worker threads,
# because the app copies the context into the thread that runs generation).
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("neuroserve_request_id", default=None)

# Attributes every LogRecord has; anything else on a record came from `extra=` and is logged as a field.
_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime", "request_id", "color_message"}


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                out[key] = value
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s")


def configure_logging(level: str = "INFO", fmt: str = "json", stream: TextIO | None = None) -> None:
    """Sends all logging (NeuroServe and uvicorn) through one handler in the chosen format."""
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error"):
        uv = logging.getLogger(name)
        uv.handlers.clear()
        uv.propagate = True
    logging.getLogger("uvicorn.access").disabled = True  # NeuroServe writes its own access log
