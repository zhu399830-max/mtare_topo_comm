"""V1R3 must change only run identity and preserve V1R2 execution hooks."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_aee_domain_sensor_export_v1r3",
        ROOT / "tools/v3/run_aee_domain_sensor_export_v1r3.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v1r3_changes_only_run_identity_and_reuses_v1r2_hooks(monkeypatch) -> None:
    runner = _load()
    calls = []
    monkeypatch.setattr(runner, "validate_authorization_timestamp", lambda path: calls.append("authorization"))
    monkeypatch.setattr(runner.sys, "argv", ["runner", "--spec", "/tmp/spec.json"])
    monkeypatch.setattr(runner.environment, "verify_host_python", lambda: calls.append("verify"))
    monkeypatch.setattr(runner.implementation, "main", lambda: calls.append("base-main") or 19)
    result = runner.main()
    assert result == 19
    assert calls == ["authorization", "verify", "base-main"]
    assert runner.implementation.RUN_ID == runner.RUN_ID
    assert runner.implementation.case_command is runner.permission_replacement.case_command
    assert runner.implementation.host_archive is runner.permission_replacement.host_archive


def test_v1r3_authorization_timestamp_accepts_prior_and_equal(tmp_path: Path) -> None:
    runner = _load()
    spec = tmp_path / "spec.json"
    for approved in ("2026-08-20T20:59:59+08:00", "2026-08-20T21:00:00+08:00"):
        spec.write_text(json.dumps({"user_authorization": {"status": "APPROVED", "approved_at": approved}}))
        result = runner.validate_authorization_timestamp(
            spec, datetime.fromisoformat("2026-08-20T21:00:00+08:00")
        )
        assert result.isoformat() == approved


def test_v1r3_authorization_timestamp_rejects_future_missing_timezone_and_invalid(tmp_path: Path) -> None:
    import pytest

    runner = _load()
    spec = tmp_path / "spec.json"
    cases = (
        ({"status": "APPROVED", "approved_at": "2026-08-20T21:00:01+08:00"}, "later than"),
        ({"status": "APPROVED", "approved_at": "2026-08-20T21:00:00"}, "timezone"),
        ({"status": "APPROVED", "approved_at": "invalid"}, "ISO-8601"),
        ({"status": "PENDING_REPLACEMENT_APPROVAL"}, "explicit"),
    )
    for authorization, message in cases:
        spec.write_text(json.dumps({"user_authorization": authorization}))
        with pytest.raises(RuntimeError, match=message):
            runner.validate_authorization_timestamp(
                spec, datetime.fromisoformat("2026-08-20T21:00:00+08:00")
            )
