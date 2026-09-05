from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mtare_topo.evaluation.stochastic_closed_loop import METRICS


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_corrected_stochastic_comparison_v1",
        ROOT / "tools/v3/run_corrected_stochastic_comparison_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_sealed_source(runner, root: Path) -> tuple[dict, Path]:
    source = root / "source"
    (source / "config").mkdir(parents=True)
    (source / "metrics").mkdir()
    (source / "artifacts/cases/000_case").mkdir(parents=True)
    state_path = source / "RUN_STATE.json"
    run_summary_path = source / "metrics/summary.json"
    schedule_path = source / "config/case_schedule.json"
    case_summary_path = source / "artifacts/cases/000_case/summary.json"
    scheduled = {
        "index": 0,
        "case_id": "000_case",
        "block_id": "tunnel_env11",
        "method_family": "m1d_topology",
        "method_id": "m1d_seed0",
        "world": "tunnel",
        "environment_seed": 11,
        "execution_repeat": 0,
        "checkpoint_seed": 0,
        "runtime_sec": 600,
    }
    metrics = {metric: float(index + 1) for index, metric in enumerate(METRICS)}
    metrics.update(
        final_pose_xyz_m=[1.0, 2.0, 3.0],
        final_waypoint_xyz_m=[4.0, 5.0, 6.0],
        recording_audit={"passed": True},
    )
    case_summary = {
        "schema_version": "mtare_single_robot_case_summary_v1",
        "status": "PASS_SINGLE_ROBOT_CASE_V2",
        "case": {"schema_version": "mtare_single_robot_case_contract_v1", **scheduled},
        "metrics": metrics,
        "planner_evidence": {"kind": "m1d_topology", "failed_cycles": 0},
        "method_identity": {"method_family": "m1d_topology"},
        "storage": {
            "archive_sha256": "a" * 64,
            "decompressed_sha256": "b" * 64,
            "original_bag_sha256": "b" * 64,
        },
    }
    state_path.write_text(
        json.dumps({"state": "COMPLETED", "overall_status": "PASS_SOURCE"}) + "\n",
        encoding="utf-8",
    )
    run_summary_path.write_text(
        json.dumps({"completed_case_count": 1}) + "\n", encoding="utf-8"
    )
    schedule_path.write_text(json.dumps({"cases": [scheduled]}) + "\n", encoding="utf-8")
    case_summary_path.write_text(json.dumps(case_summary) + "\n", encoding="utf-8")
    seal_path = source / "artifacts/evidence_sha256.txt"
    bound = (state_path, run_summary_path, schedule_path, case_summary_path)
    seal_path.write_text(
        "".join(
            f"{runner.sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in bound
        ),
        encoding="utf-8",
    )
    identity = {
        "run": "source",
        "expected_state": "COMPLETED",
        "expected_status": "PASS_SOURCE",
        "seal_sha256": runner.sha256(seal_path),
        "schedule_relative_path": "config/case_schedule.json",
        "schedule_file_sha256": runner.sha256(schedule_path),
        "schedule_content_sha256": runner.schedule_content_sha256([scheduled]),
    }
    return identity, case_summary_path


def test_load_sealed_cases_checks_full_contract_and_manifest(tmp_path: Path) -> None:
    runner = _load()
    runner.PROJECT_ROOT = tmp_path
    identity, _ = _build_sealed_source(runner, tmp_path)
    summaries, manifest = runner.load_sealed_cases(identity, expected_count=1)
    assert len(summaries) == 1
    assert manifest["case_count"] == 1
    assert manifest["verified_bound_file_count"] == 4
    assert manifest["cases"][0]["case_id"] == "000_case"
    assert manifest["schedule_file_sha256"] == identity["schedule_file_sha256"]


def test_load_sealed_cases_rejects_post_seal_summary_mutation(tmp_path: Path) -> None:
    runner = _load()
    runner.PROJECT_ROOT = tmp_path
    identity, case_summary_path = _build_sealed_source(runner, tmp_path)
    case_summary_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash drift"):
        runner.load_sealed_cases(identity, expected_count=1)
