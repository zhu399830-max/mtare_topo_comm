from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_aee_composite_v9_combined_correction_probe_v1",
        ROOT / "tools/v3/run_aee_composite_v9_combined_correction_probe_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selected():
    return {
        "case_id": "052_tunnel_env11_m1d_seed0_repeat0",
        "world": "tunnel",
        "environment_seed": 11,
        "checkpoint_seed": 0,
        "first_reanchor_proxy_frame": 144,
        "first_nonmatching_frontier_event_frame": 119,
        "both_mechanisms_observed_by_frame": 144,
    }


def _probe_case():
    return {
        "case_id": "tunnel_env11_m1d_seed0_combined_v5_probe",
        "source_case_id": "052_tunnel_env11_m1d_seed0_repeat0",
        "world": "tunnel",
        "environment_seed": 11,
        "checkpoint_seed": 0,
        "runtime_sec": 180.0,
    }


def test_probe_identity_and_command_use_only_v5_case_wrapper(tmp_path: Path):
    module = _load()
    assert module.validate_probe_case(_probe_case(), _selected()) == _probe_case()
    command, name = module.case_command(
        tmp_path, _probe_case(), {"path": "checkpoint.pt", "sha256": "a" * 64},
        "topics.json",
    )
    assert name == "aee-v5-combined-correction-probe"
    assert "/workspace/tools/v3/run_mtare_single_robot_case_v3.py" in command[-1]
    assert "--environment-seed 11" in command[-1]
    assert "--checkpoint-seed 0" in command[-1]


def _mechanism_case(tmp_path: Path, *, correction: bool = True) -> Path:
    case = tmp_path / "case"
    planner = case / "planner"
    planner.mkdir(parents=True)
    (case / "summary.json").write_text(json.dumps({
        "planner_evidence": {
            "topology_snapshot": "planner/topology_snapshot.json",
            "decision_trace": "planner/decision_trace.jsonl",
        }
    }) + "\n", encoding="utf-8")
    outcomes = {"divergent_verified_departure": 1} if correction else {}
    (planner / "topology_snapshot.json").write_text(json.dumps({
        "schema_version": "semantic_topology_global_node_v5_snapshot_v1",
        "runtime": {
            "schema_version": "online_topology_planner_runtime_v3",
            "verified_reanchor_count": 0,
            "frontier_execution_rejection_count": int(correction),
            "frontier_execution_outcomes": outcomes,
        },
    }) + "\n", encoding="utf-8")
    rows = []
    for frame, arc in enumerate((0.0, 5.0, 6.0)):
        update = {"reason": "no_event"}
        if correction and frame == 1:
            update["frontier_execution_feedback"] = {
                "classified": True,
                "outcome": "divergent_verified_departure",
                "record_rejection": True,
            }
        rows.append({"frame_index": frame, "route_arc_m": arc, "graph_update": update})
    (planner / "decision_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return case


def test_combined_probe_requires_activation_and_post_correction_progress(tmp_path: Path):
    module = _load()
    audit = module.audit_combined_correction(_mechanism_case(tmp_path))
    assert audit["frontier_execution_rejection_count"] == 1
    assert audit["verified_reanchor_count"] == 0
    assert audit["first_post_correction_growth_frame"] == 2
    assert audit["post_correction_route_growth"] is True


def test_combined_probe_rejects_a_run_with_no_corrective_event(tmp_path: Path):
    module = _load()
    with pytest.raises(RuntimeError, match="did not activate"):
        module.audit_combined_correction(_mechanism_case(tmp_path, correction=False))


def test_readiness_mode_accepts_a_valid_run_that_needs_no_correction(tmp_path: Path):
    module = _load()
    audit = module.audit_combined_correction(
        _mechanism_case(tmp_path, correction=False), require_correction=False
    )
    assert audit["first_corrective_frame"] is None
    assert audit["post_correction_route_growth"] is False


def test_probe_binds_sealed_outcome_blind_audit_selection(tmp_path: Path):
    module = _load()
    source = tmp_path / "audit"
    (source / "metrics").mkdir(parents=True)
    (source / "config").mkdir()
    (source / "artifacts").mkdir()
    selected = _selected()
    files = {
        "state": source / "RUN_STATE.json",
        "summary": source / "metrics/summary.json",
        "evidence": source / "metrics/frontier_attempt_outcomes.json",
        "manifest": source / "config/source_manifest.json",
        "provenance": source / "config/source_provenance.json",
    }
    files["state"].write_text(json.dumps({
        "state": "COMPLETED", "overall_status": module.AUDIT_STATUS_PASS,
    }) + "\n")
    files["summary"].write_text(json.dumps({
        "overall_status": module.AUDIT_STATUS_PASS,
        "source_case_count": 90, "v9_case_count": 30,
    }) + "\n")
    cases = [{
        "case_id": selected["case_id"], "world": selected["world"],
        "environment_seed": selected["environment_seed"],
        "checkpoint_seed": selected["checkpoint_seed"],
    }]
    cases.extend({
        "case_id": f"case_{index}", "world": "garage",
        "environment_seed": index, "checkpoint_seed": index % 3,
    } for index in range(29))
    files["evidence"].write_text(json.dumps({
        "case_count": 30, "raw_bag_reads": 0, "gt_or_map_reads": 0,
        "cases": cases,
        "combined_correction_probe_selection": {
            "selection_rule": "minimize_max_first_mechanism_frame_then_case_id",
            "performance_outcomes_used_for_selection": [],
            "eligible_case_count": 1,
            "selected_case": selected,
            "ranked_candidates": [selected],
        },
    }) + "\n")
    files["manifest"].write_text(json.dumps({
        "case_count": 90,
        "family_case_counts": {
            "layered_gt_map_oracle": 30, "m1d_topology": 30, "original_mtare": 30,
        },
    }) + "\n")
    files["provenance"].write_text(json.dumps({"source_mutation_permitted": False}) + "\n")
    seal_path = source / "artifacts/evidence_sha256.txt"
    seal_path.write_text("".join(
        f"{module.sha256(path)}  {path.relative_to(tmp_path).as_posix()}\n"
        for path in files.values()
    ))
    spec = {
        "mechanism_audit_run": "audit",
        "mechanism_audit_status": module.AUDIT_STATUS_PASS,
        "mechanism_audit_seal_sha256": module.sha256(seal_path),
    }
    value = module.validate_composed_audit_source(spec, project_root=tmp_path)
    assert value["selected_source_case"] == selected
    assert value["performance_outcomes_used_for_selection"] == []
