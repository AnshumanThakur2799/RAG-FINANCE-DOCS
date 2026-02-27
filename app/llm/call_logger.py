from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

_LOG_FILE = Path("llm_calls.log")
_LOCK = Lock()


def log_llm_call(
    *,
    source: str,
    operation: str,
    status: str,
    query: str | None = None,
    details: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "operation": operation,
        "status": status,
    }
    if query is not None:
        payload["query"] = query
    if details:
        payload["details"] = details
    if error:
        payload["error"] = error

    line = json.dumps(payload, ensure_ascii=True)
    try:
        with _LOCK:
            _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with _LOG_FILE.open("a", encoding="utf-8") as file_obj:
                file_obj.write(f"{line}\n")
    except Exception:
        # Logging should never interrupt normal request flow.
        pass
