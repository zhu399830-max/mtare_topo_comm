from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_aee_composite_v9_reanchor_readiness_v1",
        ROOT / "tools/v3/run_aee_composite_v9_reanchor_readiness_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_six_case_readiness_replaces_only_case_wrapper_and_container_identity(tmp_path: Path) -> None:
    runner = _load()
    runner.ACTIVE_SPEC = {}
    runner.validate_probe_source = lambda spec: {
        "schema_version": "aee_composite_v9_reanchor_probe_source_v1"
    }
    (tmp_path / "config").mkdir()
    case = {
        "case_id": "tunnel_v9_seed0_env11",
        "world": "tunnel",
        "environment_seed": 11,
        "checkpoint_seed": 0,
        "runtime_sec": 180.0,
    }
    command, name = runner.case_command(
        tmp_path,
        case,
        {"path": "checkpoint.pt", "sha256": "a" * 64},
        "configs/v3/gate6/closed_loop_recording_topics_aee_v2.json",
    )
    assert name == "aee-v4-readiness-tunnel_v9_seed0_env11"
    assert command[command.index("--name") + 1] == name
    assert "/workspace/tools/v3/run_mtare_single_robot_case_v2.py" in command[-1]
    assert "/workspace/tools/v3/run_mtare_single_robot_case_v1.py" not in command[-1]
    assert (tmp_path / "config/reanchor_probe_source.json").is_file()


def test_readiness_requires_exact_sealed_probe_with_causal_progress(tmp_path: Path) -> None:
    runner = _load()
    source = tmp_path / "probe"
    (source / "metrics").mkdir(parents=True)
    (source / "artifacts/cases/probe_case").mkdir(parents=True)
    state_path = source / "RUN_STATE.json"
    summary_path = source / "metrics/summary.json"
    case_path = source / "artifacts/cases/probe_case/summary.json"
    state_path.write_text(
        json.dumps({"state": "COMPLETED", "overall_status": "PASS_PROBE"}) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps({
            "overall_status": "PASS_PROBE",
            "completed_cases": 1,
            "mechanism_audit": {
                "verified_reanchor_count": 1,
                "post_reanchor_route_growth": True,
                "uses_evaluator_gt": False,
            },
        }) + "\n",
        encoding="utf-8",
    )
    case_path.write_text(
        json.dumps({"status": "PASS_SINGLE_ROBOT_CASE_V2"}) + "\n", encoding="utf-8"
    )
    seal_path = source / "artifacts/evidence_sha256.txt"
    seal_path.write_text(
        "".join(
            f"{runner.sha256(path)}  {path.relative_to(tmp_path).as_posix()}\n"
            for path in (state_path, summary_path, case_path)
        ),
        encoding="utf-8",
    )
    spec = {
        "reanchor_probe_run": "probe",
        "reanchor_probe_case_id": "probe_case",
        "reanchor_probe_status": "PASS_PROBE",
        "reanchor_probe_seal_sha256": runner.sha256(seal_path),
    }
    evidence = runner.validate_probe_source(spec, project_root=tmp_path)
    assert evidence["verified_reanchor_count"] == 1
    assert evidence["post_reanchor_route_growth"] is True
