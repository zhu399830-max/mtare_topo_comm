from __future__ import annotations

import importlib.util
import json
import sys
import pytest
from pathlib import Path

from mtare_topo.evaluation.closed_loop_matrix import (
    enumerate_stochastic_cases,
    load_stochastic_matrix,
)


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_aee_composite_v9_combined_correction_stochastic_v1",
        ROOT / "tools/v3/run_aee_composite_v9_combined_correction_stochastic_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cases():
    matrix = load_stochastic_matrix(ROOT / "configs/v3/gate6/mtare_single_robot_stochastic_matrix_v3.json")
    all_cases = list(enumerate_stochastic_cases(matrix))
    return matrix, all_cases, [case for case in all_cases if case.method_family == "m1d_topology"]


def test_v5_case_command_uses_combined_case_wrapper(tmp_path: Path):
    module = _load()
    matrix, _, selected = _cases()
    command, name = module.case_command(tmp_path, selected[0], matrix)
    assert name.startswith("mtare-v9-v5-")
    assert command[command.index("--name") + 1] == name
    assert "/workspace/tools/v3/run_mtare_single_robot_case_v3.py" in command[-1]
    assert "/workspace/tools/v3/run_mtare_single_robot_case_v1.py" not in command[-1]


def test_v5_schedule_binds_exact_composed_audit_manifest(tmp_path: Path):
    module = _load()
    _, all_cases, selected = _cases()
    source = tmp_path / "audit"
    (source / "config").mkdir(parents=True)
    (source / "metrics").mkdir()
    (source / "artifacts").mkdir()
    paths = [
        source / "RUN_STATE.json", source / "metrics/summary.json",
        source / "config/source_manifest.json", source / "config/source_provenance.json",
    ]
    paths[0].write_text(json.dumps({"state": "COMPLETED", "overall_status": module.EXPECTED_AUDIT_STATUS}) + "\n")
    paths[1].write_text(json.dumps({
        "overall_status": module.EXPECTED_AUDIT_STATUS,
        "source_case_count": 90, "v9_case_count": 30,
    }) + "\n")
    manifest_cases = [{
        "index": case.index, "case_id": case.case_id, "block_id": case.block_id,
        "method_family": case.method_family,
        "source": "recovery_run" if case.block_id == "tunnel_env23" or case.index >= 77 else "failed_source_run",
    } for case in all_cases]
    paths[2].write_text(json.dumps({"case_count": 90, "cases": manifest_cases}) + "\n")
    paths[3].write_text(json.dumps({"source_mutation_permitted": False}) + "\n")
    seal_path = source / "artifacts/evidence_sha256.txt"
    seal_path.write_text("".join(
        f"{module.base.matrix_base.sha256(path)}  {path.relative_to(tmp_path).as_posix()}\n"
        for path in paths
    ))
    spec = {
        "paired_source_audit_run": "audit",
        "paired_source_audit_status": module.EXPECTED_AUDIT_STATUS,
        "paired_source_audit_seal_sha256": module.base.matrix_base.sha256(seal_path),
    }
    evidence = module.validate_composed_paired_source_schedule(
        spec, selected, project_root=tmp_path
    )
    assert evidence["matched_m1d_case_count"] == 30
    assert evidence["exact_order_and_identity_match"] is True
    assert evidence["source_schedule_path"].endswith("config/source_manifest.json")
    assert len(evidence["source_schedule_file_sha256"]) == 64
    assert len(evidence["source_schedule_content_sha256"]) == 64
    assert sum(evidence["m1d_source_counts"].values()) == 30
    assert evidence["m1d_source_counts"] == {
        "failed_source_run": 22,
        "recovery_run": 8,
    }


def test_v5_schedule_rejects_wrong_composed_source_quota(tmp_path: Path):
    module = _load()
    _, all_cases, selected = _cases()
    source = tmp_path / "audit"
    (source / "config").mkdir(parents=True)
    (source / "metrics").mkdir()
    (source / "artifacts").mkdir()
    paths = [
        source / "RUN_STATE.json", source / "metrics/summary.json",
        source / "config/source_manifest.json", source / "config/source_provenance.json",
    ]
    paths[0].write_text(json.dumps({
        "state": "COMPLETED", "overall_status": module.EXPECTED_AUDIT_STATUS,
    }) + "\n")
    paths[1].write_text(json.dumps({
        "overall_status": module.EXPECTED_AUDIT_STATUS,
        "source_case_count": 90, "v9_case_count": 30,
    }) + "\n")
    manifest_cases = [{
        "index": case.index, "case_id": case.case_id, "block_id": case.block_id,
        "method_family": case.method_family,
        "source": "failed_source_run",
    } for case in all_cases]
    paths[2].write_text(json.dumps({"case_count": 90, "cases": manifest_cases}) + "\n")
    paths[3].write_text(json.dumps({"source_mutation_permitted": False}) + "\n")
    seal_path = source / "artifacts/evidence_sha256.txt"
    seal_path.write_text("".join(
        f"{module.base.matrix_base.sha256(path)}  {path.relative_to(tmp_path).as_posix()}\n"
        for path in paths
    ))
    spec = {
        "paired_source_audit_run": "audit",
        "paired_source_audit_status": module.EXPECTED_AUDIT_STATUS,
        "paired_source_audit_seal_sha256": module.base.matrix_base.sha256(seal_path),
    }
    with pytest.raises(RuntimeError, match="composition drift"):
        module.validate_composed_paired_source_schedule(
            spec, selected, project_root=tmp_path
        )


def test_v5_mechanism_summary_counts_both_corrections(tmp_path: Path):
    module = _load()
    summaries = []
    for index, (reanchor_count, rejection_count) in enumerate(((1, 0), (0, 2), (3, 4))):
        case_id = f"case_{index}"
        planner = tmp_path / f"artifacts/cases/{case_id}/planner"
        planner.mkdir(parents=True)
        snapshot = planner / "snapshot.json"
        snapshot.write_text(json.dumps({"runtime": {
            "verified_reanchor_count": reanchor_count,
            "frontier_execution_rejection_count": rejection_count,
            "frontier_execution_outcomes": {"divergent_verified_departure": rejection_count},
        }}) + "\n")
        summaries.append({
            "case": {"case_id": case_id},
            "planner_evidence": {"topology_snapshot": "planner/snapshot.json"},
        })
    value = module.summarize_mechanisms(summaries, tmp_path)
    assert value["verified_reanchor_count_total"] == 4
    assert value["frontier_execution_rejection_count_total"] == 6
    assert value["case_count_with_both_correction_types"] == 1
