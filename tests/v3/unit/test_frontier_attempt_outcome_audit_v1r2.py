from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import platform
import shlex
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_frontier_attempt_outcome_audit_v1r2",
        ROOT / "tools/v3/run_frontier_attempt_outcome_audit_v1r2.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _material_run(root: Path):
    run_id = "gate6_20260823_frontier_attempt_v1r2_test_seed0"
    spec = {
        "run_id": run_id,
        "config_path": "configs/spec.json",
        "data_card": "configs/card.json",
        "command": [sys.executable, "runner.py", "--run", run_id],
    }
    (root / "configs").mkdir()
    (root / "results").mkdir()
    spec_path, card_path = root / spec["config_path"], root / spec["data_card"]
    spec_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    card_path.write_text(json.dumps({"card": "exact"}) + "\n", encoding="utf-8")
    status = {"current_phase": 6, "current_gate": 6}
    (root / "results/project_status.json").write_text(json.dumps(status) + "\n", encoding="utf-8")
    run_dir = root / "results/gate6_single_robot" / run_id
    for child in ("config", "logs", "metrics", "previews", "artifacts"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    (run_dir / "config/source_config").write_bytes(spec_path.read_bytes())
    environment = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "platform": platform.platform(),
        "python": sys.version, "python_executable": sys.executable,
    }
    for name, value in (
        ("run_spec.json", spec), ("data_card.json", {"card": "exact"}),
        ("status_snapshot.json", status), ("environment.json", environment),
    ):
        (run_dir / "config" / name).write_text(json.dumps(value) + "\n", encoding="utf-8")
    (run_dir / "config/command.txt").write_text(shlex.join(spec["command"]) + "\n", encoding="utf-8")
    (run_dir / "RUN_STATE.json").write_text(json.dumps({
        "run_id": run_id, "state": "CREATED_NOT_EXECUTED"
    }) + "\n", encoding="utf-8")
    return spec_path, spec, run_dir


def test_v1r2_material_identity_binds_status_and_environment(tmp_path):
    module = _load()
    spec_path, spec, run_dir = _material_run(tmp_path)
    module.validate_material_run_identity_v1r2(
        spec_path, spec, run_dir, project_root=tmp_path
    )
    (run_dir / "config/status_snapshot.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="project-status snapshot drift"):
        module.validate_material_run_identity_v1r2(
            spec_path, spec, run_dir, project_root=tmp_path
        )


def test_v1r2_material_identity_rejects_environment_content_drift(tmp_path):
    module = _load()
    spec_path, spec, run_dir = _material_run(tmp_path)
    environment = json.loads((run_dir / "config/environment.json").read_text())
    environment["hostname"] = "forged"
    (run_dir / "config/environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="environment identity drift"):
        module.validate_material_run_identity_v1r2(
            spec_path, spec, run_dir, project_root=tmp_path
        )


def test_v1r2_source_generator_identity_binds_all_producers(tmp_path):
    module = _load()
    identities = {}
    for name in ("causal_graph", "frontier_planner", "online_runtime", "v9_node"):
        path = tmp_path / f"{name}.py"
        path.write_text(f"# {name}\n", encoding="utf-8")
        identities[name] = {"path": path.name, "sha256": module.sha256(path)}
    source = tmp_path / "source/config"
    source.mkdir(parents=True)
    (source / "run_spec.json").write_text(json.dumps({
        "frozen_tools": identities
    }) + "\n", encoding="utf-8")
    spec = {"source_run": "source"}
    for name, identity in identities.items():
        spec[f"source_{name}_path"] = identity["path"]
        spec[f"source_{name}_sha256"] = identity["sha256"]
    assert module.validate_source_generator_identity(
        spec, project_root=tmp_path
    ) == identities
    spec["source_v9_node_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="v9_node"):
        module.validate_source_generator_identity(spec, project_root=tmp_path)
