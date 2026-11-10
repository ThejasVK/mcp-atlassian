"""Tests for the hash-chained tool-call audit log."""

import json
import threading

import pytest

from mcp_atlassian.utils import audit
from mcp_atlassian.utils.audit import (
    GENESIS_HASH,
    REDACTED,
    AuditLogger,
    get_audit_logger,
    redact,
    reset_audit_logger,
    verify_audit_log,
)


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_lines(path, entries):
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))


class TestRedact:
    def test_masks_secret_keys_at_any_depth(self):
        out = redact(
            {
                "summary": "ok",
                "api_token": "abc",
                "headers": {"Authorization": "Bearer x", "X-Trace": "1"},
                "items": [{"password": "p"}],
            }
        )
        assert out["summary"] == "ok"
        assert out["api_token"] == REDACTED
        assert out["headers"] == {"Authorization": REDACTED, "X-Trace": "1"}
        assert out["items"] == [{"password": REDACTED}]

    def test_trims_long_strings_and_lists(self):
        out = redact({"body": "x" * 5000, "keys": list(range(50))})
        assert out["body"].endswith("<5000 chars>")
        assert len(out["body"]) < 300
        assert len(out["keys"]) == audit.MAX_LIST_ITEMS + 1
        assert out["keys"][-1] == "<30 more items>"

    def test_non_json_values_become_strings(self):
        assert redact({"when": object})["when"].startswith("<class")


class TestAuditLogger:
    def test_records_form_a_valid_chain(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        a = AuditLogger(log)
        a.record(
            tool="jira_get_issue",
            arguments={"issue_key": "PROJ-1"},
            outcome="ok",
            duration_ms=12.34,
        )
        a.record(
            tool="jira_create_issue",
            arguments={"summary": "New"},
            outcome="ok",
            duration_ms=40,
            write=True,
            dry_run=True,
        )

        entries = _read(log)
        assert entries[0]["prev_hash"] == GENESIS_HASH
        assert entries[1]["prev_hash"] == entries[0]["hash"]
        assert entries[1]["write"] is True and entries[1]["dry_run"] is True
        assert entries[0]["duration_ms"] == 12.3
        assert verify_audit_log(log) == (True, 2, "OK: 2 records, chain intact")

    def test_secrets_never_reach_the_file(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        AuditLogger(log).record(
            tool="jira_add_comment",
            arguments={"comment": "hi", "personal_token": "SUPERSECRET"},
            outcome="error",
            duration_ms=1,
            error="401 Unauthorized",
        )
        text = log.read_text()
        assert "SUPERSECRET" not in text
        assert _read(log)[0]["error"] == "401 Unauthorized"

    def test_continues_chain_across_restarts(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        AuditLogger(log).record(tool="a", arguments={}, outcome="ok", duration_ms=1)
        AuditLogger(log).record(tool="b", arguments={}, outcome="ok", duration_ms=1)
        assert verify_audit_log(log)[0] is True
        assert len(_read(log)) == 2

    def test_refuses_to_extend_a_corrupt_log(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        log.write_text("not json\n")
        with pytest.raises(ValueError, match="unreadable record"):
            AuditLogger(log)

    def test_thread_safe_appends(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        a = AuditLogger(log)

        def worker(n):
            for i in range(25):
                a.record(tool=f"t{n}", arguments={"i": i}, outcome="ok", duration_ms=0)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert verify_audit_log(log) == (True, 100, "OK: 100 records, chain intact")


class TestVerify:
    def _two_records(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        a = AuditLogger(log)
        a.record(tool="one", arguments={"k": 1}, outcome="ok", duration_ms=1)
        a.record(tool="two", arguments={"k": 2}, outcome="ok", duration_ms=1)
        return log, _read(log)

    def test_detects_edited_record(self, tmp_path):
        log, entries = self._two_records(tmp_path)
        entries[0]["arguments"]["k"] = 99
        _write_lines(log, entries)
        ok, line, msg = verify_audit_log(log)
        assert (ok, line) == (False, 1)
        assert "modified" in msg

    def test_detects_deleted_record(self, tmp_path):
        log, entries = self._two_records(tmp_path)
        _write_lines(log, entries[1:])
        ok, line, msg = verify_audit_log(log)
        assert (ok, line) == (False, 1)
        assert "chain broken" in msg

    def test_detects_reordered_records(self, tmp_path):
        log, entries = self._two_records(tmp_path)
        _write_lines(log, list(reversed(entries)))
        assert verify_audit_log(log)[0] is False

    def test_missing_file(self, tmp_path):
        assert verify_audit_log(tmp_path / "nope.jsonl")[0] is False

    def test_cli_exit_codes(self, tmp_path, capsys):
        log, entries = self._two_records(tmp_path)
        assert audit._main(["verify", str(log)]) == 0
        entries[1]["tool"] = "tampered"
        _write_lines(log, entries)
        assert audit._main(["verify", str(log)]) == 1
        assert audit._main([]) == 2
        assert "line 2" in capsys.readouterr().out


class TestGetAuditLogger:
    def setup_method(self):
        reset_audit_logger()

    def teardown_method(self):
        reset_audit_logger()

    def test_disabled_without_env(self, monkeypatch):
        monkeypatch.delenv(audit.AUDIT_LOG_ENV, raising=False)
        assert get_audit_logger() is None

    def test_enabled_and_cached(self, monkeypatch, tmp_path):
        monkeypatch.setenv(audit.AUDIT_LOG_ENV, str(tmp_path / "logs" / "a.jsonl"))
        first = get_audit_logger()
        assert first is not None
        assert get_audit_logger() is first
        assert (tmp_path / "logs").is_dir()

    def test_new_path_gets_new_logger(self, monkeypatch, tmp_path):
        monkeypatch.setenv(audit.AUDIT_LOG_ENV, str(tmp_path / "a.jsonl"))
        first = get_audit_logger()
        monkeypatch.setenv(audit.AUDIT_LOG_ENV, str(tmp_path / "b.jsonl"))
        assert get_audit_logger() is not first
