from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
import sys
from pathlib import Path

from test_frontier_attempt_outcome_audit_v1r2 import _material_run


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_frontier_attempt_outcome_audit_v1r3",
        ROOT / "tools/v3/run_frontier_attempt_outcome_audit_v1r3.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v1r3_accepts_exact_but_old_create_run_environment(tmp_path):
    module = _load()
    spec_path, spec, run_dir = _material_run(tmp_path)
    environment_path = run_dir / "config/environment.json"
    environment = json.loads(environment_path.read_text())
    environment["created_at_utc"] = (
        datetime.now(timezone.utc) - timedelta(days=2)
    ).isoformat()
    environment_path.write_text(json.dumps(environment) + "\n", encoding="utf-8")
    module.validate_material_run_identity_v1r3(
        spec_path, spec, run_dir, project_root=tmp_path
    )


def test_v1r3_rejects_future_environment_time(tmp_path):
    module = _load()
    spec_path, spec, run_dir = _material_run(tmp_path)
    environment_path = run_dir / "config/environment.json"
    environment = json.loads(environment_path.read_text())
    environment["created_at_utc"] = (
        datetime.now(timezone.utc) + timedelta(days=1)
    ).isoformat()
    environment_path.write_text(json.dumps(environment) + "\n", encoding="utf-8")
    try:
        module.validate_material_run_identity_v1r3(
            spec_path, spec, run_dir, project_root=tmp_path
        )
    except RuntimeError as exc:
        assert "environment identity drift" in str(exc)
    else:
        raise AssertionError("future environment timestamp was accepted")
