from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mtare_topo.evaluation.closed_loop_matrix import (
    enumerate_stochastic_cases,
    load_stochastic_matrix,
)
from mtare_topo.evaluation.stochastic_closed_loop import METRICS


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_composed_frontier_attempt_audit_v1",
        ROOT / "tools/v3/run_composed_frontier_attempt_audit_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case_summary(scheduled: dict) -> dict:
    metrics = {metric: float(index + 1) for index, metric in enumerate(METRICS)}
    metrics.update(
        final_pose_xyz_m=[1.0, 2.0, 3.0],
        final_waypoint_xyz_m=[4.0, 5.0, 6.0],
        recording_audit={"passed": True},
    )
    family = scheduled["method_family"]
    case = dict(scheduled)
    case["schema_version"] = "mtare_single_robot_case_contract_v1"
    return {
        "schema_version": "mtare_single_robot_case_summary_v1",
        "status": "PASS_SINGLE_ROBOT_CASE_V2",
        "case": case,
        "metrics": metrics,
        "planner_evidence": {
            "kind": "original_mtare_internal" if family == "original_mtare" else family,
            "failed_cycles": None if family == "original_mtare" else 0,
        },
        "method_identity": {"method_family": family},
        "storage": {
            "archive_sha256": "a" * 64,
            "decompressed_sha256": "b" * 64,
            "original_bag_sha256": "b" * 64,
        },
    }


def _seal(module, root: Path, paths: list[Path]) -> str:
    seal = root / "artifacts/evidence_sha256.txt"
    seal.parent.mkdir(parents=True, exist_ok=True)
    seal.write_text(
        "".join(
            f"{module.sha256(path)}  {path.relative_to(root.parent).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
    return module.sha256(seal)


def _sources(tmp_path: Path):
    module = _load()
    matrix = load_stochastic_matrix(ROOT / "configs/v3/gate6/mtare_single_robot_stochastic_matrix_v3.json")
    schedule = [case.to_dict() for case in enumerate_stochastic_cases(matrix)]
    predecessor, recovery = tmp_path / "predecessor", tmp_path / "recovery"
    for root in (predecessor, recovery):
        (root / "metrics").mkdir(parents=True)
        (root / "config").mkdir()
        (root / "artifacts/cases").mkdir(parents=True)
    predecessor_state = predecessor / "RUN_STATE.json"
    predecessor_state.write_text(json.dumps({"state": "FAILED", "overall_status": module.EXPECTED_PREDECESSOR_STATUS}) + "\n")
    predecessor_summary = predecessor / "metrics/summary.json"
    predecessor_summary.write_text(json.dumps({"overall_status": module.EXPECTED_PREDECESSOR_STATUS, "completed_case_count": 77}) + "\n")
    predecessor_spec = predecessor / "config/run_spec.json"
    predecessor_spec.write_text(json.dumps({"frozen_tools": {}}) + "\n")
    schedule_path = predecessor / "config/case_schedule.json"
    schedule_path.write_text(json.dumps({"cases": schedule}, sort_keys=True) + "\n")

    recovery_state = recovery / "RUN_STATE.json"
    recovery_state.write_text(json.dumps({"state": "COMPLETED", "overall_status": module.EXPECTED_RECOVERY_STATUS}) + "\n")
    recovery_summary = recovery / "metrics/summary.json"
    recovery_summary.write_text(json.dumps({
        "overall_status": module.EXPECTED_RECOVERY_STATUS,
        "completed_recovery_case_count": 21,
        "combined_case_count": 90,
        "source_reused_case_count": 69,
    }) + "\n")
    recovery_spec = recovery / "config/run_spec.json"
    recovery_spec.write_text(json.dumps({"replacement_contract": {"scientific_contract_change": False}, "frozen_tools": {}}) + "\n")

    mapping = []
    predecessor_paths = [predecessor_state, predecessor_summary, predecessor_spec, schedule_path]
    recovery_paths = [recovery_state, recovery_summary, recovery_spec]
    for scheduled in schedule:
        source = "recovery_run" if scheduled["block_id"] == "tunnel_env23" or scheduled["index"] >= 77 else "failed_source_run"
        root = recovery if source == "recovery_run" else predecessor
        case_dir = root / "artifacts/cases" / scheduled["case_id"]
        case_dir.mkdir()
        summary_path = case_dir / "summary.json"
        summary_path.write_text(json.dumps(_case_summary(scheduled), sort_keys=True) + "\n")
        (recovery_paths if source == "recovery_run" else predecessor_paths).append(summary_path)
        mapping.append({"case_id": scheduled["case_id"], "original_schedule_index": scheduled["index"], "source": source})
    map_path = recovery / "config/combined_case_sources.json"
    map_path.write_text(json.dumps({"cases": mapping}, sort_keys=True) + "\n")
    recovery_paths.append(map_path)
    predecessor_seal = _seal(module, predecessor, predecessor_paths)
    recovery_seal = _seal(module, recovery, recovery_paths)
    spec = {
        "predecessor_run": "predecessor",
        "predecessor_seal_sha256": predecessor_seal,
        "source_run": "recovery",
        "source_seal_sha256": recovery_seal,
        "source_schedule_file_sha256": module.sha256(schedule_path),
        "source_schedule_content_sha256": module.schedule_content_sha256(schedule),
    }
    return module, spec, schedule, mapping


def test_composed_loader_accepts_exact_69_plus_21_source(tmp_path: Path):
    module, spec, schedule, _ = _sources(tmp_path)
    summaries, roots, seals, source_specs, source = module.load_composed_source(spec, project_root=tmp_path)
    assert len(summaries) == len(roots) == 90
    assert source["manifest"]["source_counts"] == {"failed_source_run": 69, "recovery_run": 21}
    assert sum(row["block_id"] == "tunnel_env23" and row["source"] == "recovery_run" for row in source["manifest"]["cases"]) == 9
    assert set(seals) == set(source_specs) == {"failed_source_run", "recovery_run"}
    assert [item["case"]["case_id"] for item in summaries] == [item["case_id"] for item in schedule]


def test_composed_source_enters_frozen_statistics_through_status_only_bridge(tmp_path: Path):
    module, spec, _, _ = _sources(tmp_path)
    summaries, _, _, _, _ = module.load_composed_source(spec, project_root=tmp_path)
    before = [item["status"] for item in summaries]
    analysis = module.analyze_finalized_v2_cases(summaries)
    assert analysis["case_count"] == 90
    assert analysis["block_count"] == 10
    assert set(analysis["m1d_comparisons"]) == set(METRICS)
    assert analysis["compatibility_bridge"]["mapped_field"] == "status"
    assert analysis["compatibility_bridge"]["mapped_field_count"] == 90
    assert [item["status"] for item in summaries] == before


def test_composed_loader_rejects_selective_failed_case_mapping(tmp_path: Path):
    module, spec, _, mapping = _sources(tmp_path)
    recovery = tmp_path / "recovery"
    map_path = recovery / "config/combined_case_sources.json"
    first = next(row for row in mapping if row["source"] == "recovery_run" and row["original_schedule_index"] < 77)
    first["source"] = "failed_source_run"
    map_path.write_text(json.dumps({"cases": mapping}, sort_keys=True) + "\n")
    recovery_paths = [
        recovery / "RUN_STATE.json", recovery / "metrics/summary.json",
        recovery / "config/run_spec.json", map_path,
        *sorted((recovery / "artifacts/cases").glob("*/summary.json")),
    ]
    spec["source_seal_sha256"] = _seal(module, recovery, recovery_paths)
    with pytest.raises(RuntimeError, match="quota drift|source rule drift"):
        module.load_composed_source(spec, project_root=tmp_path)


def test_composed_loader_rejects_original_schedule_index_drift(tmp_path: Path):
    module, spec, _, mapping = _sources(tmp_path)
    recovery = tmp_path / "recovery"
    map_path = recovery / "config/combined_case_sources.json"
    mapping[0]["original_schedule_index"] = mapping[0]["original_schedule_index"] + 1
    map_path.write_text(json.dumps({"cases": mapping}, sort_keys=True) + "\n")
    recovery_paths = [
        recovery / "RUN_STATE.json", recovery / "metrics/summary.json",
        recovery / "config/run_spec.json", map_path,
        *sorted((recovery / "artifacts/cases").glob("*/summary.json")),
    ]
    spec["source_seal_sha256"] = _seal(module, recovery, recovery_paths)
    with pytest.raises(RuntimeError, match="original schedule index drift"):
        module.load_composed_source(spec, project_root=tmp_path)


def test_finalizer_materializes_only_after_exact_composed_source_seals(tmp_path: Path):
    module, source_spec, _, _ = _sources(tmp_path)
    finalizer_spec = importlib.util.spec_from_file_location(
        "finalize_composed_frontier_attempt_audit_v1_test",
        ROOT / "tools/v3/finalize_composed_frontier_attempt_audit_v1.py",
    )
    assert finalizer_spec is not None and finalizer_spec.loader is not None
    finalizer = importlib.util.module_from_spec(finalizer_spec)
    finalizer_spec.loader.exec_module(finalizer)
    configs = tmp_path / "configs"
    configs.mkdir()
    tool = tmp_path / "tool.py"
    tool.write_text("pass\n", encoding="utf-8")
    proposal_spec = configs / "proposal.json"
    proposal_card = configs / "card.proposal.json"
    final_spec = configs / "final.json"
    final_card = configs / "card.final.json"
    proposal = {
        **source_spec,
        "status": "PENDING_FINAL_SOURCE_SEAL_NOT_EXECUTABLE",
        "source_seal_sha256": "PENDING_FINAL_SOURCE_SEAL",
        "user_authorization": {"status": "APPROVED_AFTER_EXACT_SOURCE_SEAL_BINDING"},
        "config_path": "configs/final.json",
        "data_card": "configs/card.final.json",
        "frozen_tools": {
            "data_card_proposal": {
                "path": "configs/card.proposal.json",
                "sha256": "PENDING",
            },
            "tool": {"path": "tool.py", "sha256": module.sha256(tool)},
        },
    }
    card = {
        "status": "PENDING_FINAL_SOURCE_SEAL_NOT_EXECUTABLE",
        "approval": {"status": "APPROVED_AFTER_EXACT_SOURCE_SEAL_BINDING"},
        "sources": {"recovery_seal_sha256": "PENDING_FINAL_SOURCE_SEAL"},
    }
    proposal_card.write_text(json.dumps(card, sort_keys=True) + "\n")
    proposal["frozen_tools"]["data_card_proposal"]["sha256"] = module.sha256(proposal_card)
    proposal_spec.write_text(json.dumps(proposal, sort_keys=True) + "\n")
    card_path, spec_path = finalizer.finalize(
        proposal_spec_path=proposal_spec.relative_to(tmp_path),
        proposal_card_path=proposal_card.relative_to(tmp_path),
        final_spec_path=final_spec.relative_to(tmp_path),
        final_card_path=final_card.relative_to(tmp_path),
        project_root=tmp_path,
    )
    finalized_spec = json.loads(spec_path.read_text())
    finalized_card = json.loads(card_path.read_text())
    assert finalized_spec["user_authorization"]["status"] == "APPROVED"
    assert finalized_spec["source_seal_sha256"] == source_spec["source_seal_sha256"]
    assert finalized_card["status"] == "APPROVED_FOR_ONE_FORMAL_EXECUTION"
