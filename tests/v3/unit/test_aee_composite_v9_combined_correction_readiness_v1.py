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
        "run_aee_composite_v9_combined_correction_readiness_v1",
        ROOT / "tools/v3/run_aee_composite_v9_combined_correction_readiness_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_six_case_readiness_uses_only_v5_case_wrapper(tmp_path: Path):
    module = _load()
    module.ACTIVE_SPEC = {}
    module.validate_probe_source = lambda spec: {"schema_version": "probe_source"}
    (tmp_path / "config").mkdir()
    case = {
        "case_id": "tunnel_v9_seed0_env11", "world": "tunnel",
        "environment_seed": 11, "checkpoint_seed": 0, "runtime_sec": 180.0,
    }
    command, name = module.case_command(
        tmp_path, case, {"path": "checkpoint.pt", "sha256": "a" * 64},
        "configs/v3/gate6/closed_loop_recording_topics_aee_v2.json",
    )
    assert name == "aee-v5-readiness-tunnel_v9_seed0_env11"
    assert command[command.index("--name") + 1] == name
    assert "/workspace/tools/v3/run_mtare_single_robot_case_v3.py" in command[-1]
    assert "/workspace/tools/v3/run_mtare_single_robot_case_v1.py" not in command[-1]
    assert (tmp_path / "config/combined_probe_source.json").is_file()


def test_readiness_binds_exact_sealed_combined_probe(tmp_path: Path):
    module = _load()
    source = tmp_path / "probe"
    (source / "metrics").mkdir(parents=True)
    case_dir = source / "artifacts/cases/probe_case"
    case_dir.mkdir(parents=True)
    (source / "config").mkdir()
    selected = {"case_id": "source_case", "world": "tunnel", "environment_seed": 11, "checkpoint_seed": 0}
    paths = [
        source / "RUN_STATE.json", source / "metrics/summary.json",
        case_dir / "summary.json", source / "config/mechanism_audit_source.json",
    ]
    paths[0].write_text(json.dumps({"state": "COMPLETED", "overall_status": module.EXPECTED_PROBE_STATUS}) + "\n")
    paths[1].write_text(json.dumps({
        "overall_status": module.EXPECTED_PROBE_STATUS, "completed_cases": 1,
        "source_selected_case": selected,
        "mechanism_audit": {
            "verified_reanchor_count": 0,
            "frontier_execution_rejection_count": 1,
            "post_correction_route_growth": True, "uses_evaluator_gt": False,
        },
    }) + "\n")
    paths[2].write_text(json.dumps({"status": "PASS_SINGLE_ROBOT_CASE_V2"}) + "\n")
    paths[3].write_text(json.dumps({
        "selected_source_case": selected, "performance_outcomes_used_for_selection": [],
    }) + "\n")
    seal_path = source / "artifacts/evidence_sha256.txt"
    seal_path.write_text("".join(
        f"{module.sha256(path)}  {path.relative_to(tmp_path).as_posix()}\n" for path in paths
    ))
    spec = {
        "combined_probe_run": "probe", "combined_probe_case_id": "probe_case",
        "combined_probe_status": module.EXPECTED_PROBE_STATUS,
        "combined_probe_seal_sha256": module.sha256(seal_path),
    }
    evidence = module.validate_probe_source(spec, project_root=tmp_path)
    assert evidence["corrective_event_count"] == 1
    assert evidence["source_selected_case"] == selected
