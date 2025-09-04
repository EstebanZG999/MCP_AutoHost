# MCP_AutoHost - logging utilities
from __future__ import annotations
import orjson as _orjson
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("logs/host.jsonl")

def _json_dumps(obj) -> bytes:
    return _orjson.dumps(obj, option=_orjson.OPT_NON_STR_KEYS | _orjson.OPT_SERIALIZE_NUMPY)

def log_event(kind: str, **payload):
    """Append a JSON line with timestamp + kind + payload.
    kind: 'llm_request', 'llm_response', 'mcp_request', 'mcp_response', etc.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "kind": kind,
        **payload,
    }
    with LOG_PATH.open("ab") as f:
        f.write(_json_dumps(record) + b"\n")

def tail_logs(n: int = 50) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    for line in lines[-n:]:
        try:
            out.append(_orjson.loads(line))
        except Exception:
            continue
    return out
