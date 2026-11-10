"""Tamper-evident audit log for MCP tool calls.

When ``MCP_AUDIT_LOG_FILE`` is set, every tool call the server handles is
appended to that file as one JSON object per line (JSONL). Each record carries
the SHA-256 hash of the previous record, so editing, reordering or deleting a
line breaks the chain and is caught by ``verify_audit_log``.

Arguments are redacted before they are written: values under secret-looking
keys are masked, and long strings (page bodies, descriptions) are cut down to
a preview plus their length. The log is meant to answer "which agent call
changed this ticket, and with what", not to store customer content.

Verify a log from the command line::

    python -m mcp_atlassian.utils.audit verify /var/log/mcp-audit.jsonl
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("mcp-atlassian.utils.audit")

#: Environment variable naming the JSONL file that receives audit records.
AUDIT_LOG_ENV = "MCP_AUDIT_LOG_FILE"

#: ``prev_hash`` of the first record in a fresh log.
GENESIS_HASH = "0" * 64

#: Keys whose values are always masked, matched case-insensitively as substrings.
SECRET_KEY_MARKERS = (
    "token",
    "password",
    "passwd",
    "secret",
    "authorization",
    "api_key",
    "apikey",
    "cookie",
    "credential",
)

#: Strings longer than this are replaced by a preview and their length.
MAX_STRING_CHARS = 200

#: Lists longer than this keep only their first items.
MAX_LIST_ITEMS = 20

REDACTED = "***"


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in SECRET_KEY_MARKERS)


def redact(value: Any, *, _depth: int = 0) -> Any:
    """Return a JSON-safe copy of ``value`` with secrets masked and bulk trimmed.

    Args:
        value: Tool arguments or any nested structure of dicts, lists and scalars.

    Returns:
        A structure that is safe to write to the audit log.
    """
    if _depth > 8:
        return "<nested too deep>"
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            out[key] = REDACTED if _is_secret_key(key) else redact(v, _depth=_depth + 1)
        return out
    if isinstance(value, list | tuple | set | frozenset):
        items = list(value)
        trimmed = [redact(v, _depth=_depth + 1) for v in items[:MAX_LIST_ITEMS]]
        if len(items) > MAX_LIST_ITEMS:
            trimmed.append(f"<{len(items) - MAX_LIST_ITEMS} more items>")
        return trimmed
    if isinstance(value, str):
        if len(value) > MAX_STRING_CHARS:
            return f"{value[:MAX_STRING_CHARS]}... <{len(value)} chars>"
        return value
    if value is None or isinstance(value, bool | int | float):
        return value
    return redact(str(value), _depth=_depth + 1)


def _canonical(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_record(record: Mapping[str, Any]) -> str:
    """Hash every field except ``hash`` itself (``prev_hash`` is included)."""
    body = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def _last_hash(path: Path) -> str:
    """Return the hash of the last record in ``path``, or the genesis hash."""
    if not path.exists() or path.stat().st_size == 0:
        return GENESIS_HASH
    last_line = ""
    with path.open("rb") as fh:
        for raw in fh:
            if raw.strip():
                last_line = raw.decode("utf-8")
    if not last_line:
        return GENESIS_HASH
    try:
        return str(json.loads(last_line)["hash"])
    except (json.JSONDecodeError, KeyError) as e:
        msg = (
            f"Audit log {path} ends with an unreadable record; "
            "fix or rotate the file before writing to it."
        )
        raise ValueError(msg) from e


class AuditLogger:
    """Append-only, hash-chained JSONL writer. Safe to share across threads."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._prev_hash = _last_hash(self.path)

    def record(
        self,
        *,
        tool: str,
        arguments: Mapping[str, Any] | None,
        outcome: str,
        duration_ms: float,
        write: bool = False,
        dry_run: bool = False,
        error: str | None = None,
        actor: str | None = None,
    ) -> dict[str, Any]:
        """Append one record and return it (including its hash).

        Args:
            tool: Registered tool name, e.g. ``jira_create_issue``.
            arguments: Raw tool arguments; redacted before writing.
            outcome: ``ok``, ``error`` or ``denied``.
            duration_ms: Wall time spent in the call.
            write: True when the tool is tagged as a write tool.
            dry_run: True when the call was previewed instead of executed.
            error: Short error message when ``outcome`` is not ``ok``.
            actor: Who made the call, when the transport knows (never a token).
        """
        with self._lock:
            entry: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "tool": tool,
                "write": write,
                "dry_run": dry_run,
                "outcome": outcome,
                "duration_ms": round(duration_ms, 1),
                "arguments": redact(dict(arguments or {})),
                "prev_hash": self._prev_hash,
            }
            if actor:
                entry["actor"] = actor
            if error:
                entry["error"] = redact(error)
            entry["hash"] = _hash_record(entry)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(_canonical(entry) + "\n")
            self._prev_hash = entry["hash"]
            return entry


def verify_audit_log(path: str | os.PathLike[str]) -> tuple[bool, int, str]:
    """Check the hash chain of an audit log.

    Returns:
        ``(ok, records_checked, message)``. On failure ``records_checked`` is
        the 1-based line number of the first bad record.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return False, 0, f"{p} does not exist"
    prev = GENESIS_HASH
    count = 0
    with p.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                return False, line_no, f"line {line_no}: not valid JSON"
            if entry.get("prev_hash") != prev:
                return (
                    False,
                    line_no,
                    f"line {line_no}: chain broken (prev_hash mismatch)",
                )
            if _hash_record(entry) != entry.get("hash"):
                return False, line_no, f"line {line_no}: record was modified"
            prev = entry["hash"]
            count += 1
    return True, count, f"OK: {count} records, chain intact"


_audit_logger: AuditLogger | None = None
_audit_logger_path: str | None = None
_audit_lock = threading.Lock()


def get_audit_logger() -> AuditLogger | None:
    """Return the process-wide audit logger, or None when auditing is off."""
    global _audit_logger, _audit_logger_path
    path = os.getenv(AUDIT_LOG_ENV, "").strip()
    if not path:
        return None
    with _audit_lock:
        if _audit_logger is None or _audit_logger_path != path:
            _audit_logger = AuditLogger(path)
            _audit_logger_path = path
            logger.info(f"Audit log enabled: {_audit_logger.path}")
        return _audit_logger


def reset_audit_logger() -> None:
    """Forget the cached logger (used by tests and on config reload)."""
    global _audit_logger, _audit_logger_path
    with _audit_lock:
        _audit_logger = None
        _audit_logger_path = None


def _main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "verify":
        print("usage: python -m mcp_atlassian.utils.audit verify <audit.jsonl>")
        return 2
    ok, _, message = verify_audit_log(argv[1])
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
