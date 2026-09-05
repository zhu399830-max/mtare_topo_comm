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
        "run_aee_composite_v9_reanchor_probe_v1",
        ROOT / "tools/v3/run_aee_composite_v9_reanchor_probe_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case() -> dict:
    return {
        "case_id": "tunnel_v9_seed2_env23_reanchor_probe",
        "world": "tunnel",
        "environment_seed": 23,
        "checkpoint_seed": 2,
        "runtime_sec": 180.0,
    }


def test_probe_identity_and_command_use_only_v4_case_wrapper(tmp_path: Path) -> None:
    runner = _load()
    assert runner.validate_probe_case(_case()) == _case()
    command, name = runner.case_command(
        tmp_path,
        _case(),
        {"path": "checkpoint.pt", "sha256": "a" * 64},
        "topics.json",
    )
    assert name == "aee-v4-reanchor-probe"
    assert "/workspace/tools/v3/run_mtare_single_robot_case_v2.py" in command[-1]
    assert "--world tunnel" in command[-1]
    assert "--environment-seed 23" in command[-1]
    assert "--checkpoint-seed 2" in command[-1]


def _mechanism_case(
    tmp_path: Path, *, grow_after: bool = True, reanchor_count: int = 1
) -> Path:
    case = tmp_path / "case"
    (case / "planner").mkdir(parents=True)
    summary = {
        "planner_evidence": {
            "topology_snapshot": "planner/topology_snapshot.json",
            "decision_trace": "planner/decision_trace.jsonl",
        }
    }
    (case / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    snapshot = {
        "schema_version": "semantic_topology_global_node_v4_snapshot_v1",
        "runtime": {
            "schema_version": "online_topology_planner_runtime_v2",
            "verified_reanchor_count": reanchor_count,
            "graph": {
                "current_node": 0,
                "nodes": [{"id": 0, "xyz_m": [0.0, 0.0, 0.0]}],
            },
        },
    }
    (case / "planner/topology_snapshot.json").write_text(
        json.dumps(snapshot) + "\n", encoding="utf-8"
    )
    arcs = [0.0, 5.0, 6.0 if grow_after else 5.0]
    rows = []
    for frame, arc in enumerate(arcs):
        rows.append({
            "frame_index": frame,
            "route_arc_m": arc,
            "graph_update": {
                "node_id": 0,
                "reason": "verified_backtrack_reanchor" if frame == 1 and reanchor_count else "no_event",
            },
            "target": {"mode": "hold"},
        })
    (case / "planner/decision_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return case


def test_mechanism_audit_requires_transition_and_later_physical_progress(tmp_path: Path) -> None:
    runner = _load()
    audit = runner.audit_reanchor_mechanism(_mechanism_case(tmp_path))
    assert audit["verified_reanchor_count"] == 1
    assert audit["first_reanchor_frame"] == 1
    assert audit["first_post_reanchor_growth_frame"] == 2
    assert audit["post_reanchor_route_growth"] is True
    assert audit["uses_evaluator_gt"] is False


def test_mechanism_audit_fails_if_robot_remains_stalled(tmp_path: Path) -> None:
    runner = _load()
    with pytest.raises(RuntimeError, match="did not grow"):
        runner.audit_reanchor_mechanism(_mechanism_case(tmp_path, grow_after=False))


def test_readiness_mode_accepts_a_valid_run_that_did_not_need_reanchor(tmp_path: Path) -> None:
    runner = _load()
    audit = runner.audit_reanchor_mechanism(
        _mechanism_case(tmp_path, reanchor_count=0), require_reanchor=False
    )
    assert audit["verified_reanchor_count"] == 0
    assert audit["first_reanchor_frame"] is None
    assert audit["post_reanchor_route_growth"] is False


def test_probe_binds_exact_sealed_source_case_and_mechanism(tmp_path: Path) -> None:
    runner = _load()
    source = tmp_path / "source"
    case_id = "016_tunnel_env23_m1d_seed2_repeat0"
    case_dir = source / "artifacts/cases" / case_id
    (source / "metrics").mkdir(parents=True)
    (case_dir / "planner").mkdir(parents=True)
    state_path = source / "RUN_STATE.json"
    run_summary_path = source / "metrics/summary.json"
    case_summary_path = case_dir / "summary.json"
    trace_path = case_dir / "planner/decision_trace.jsonl"
    snapshot_path = case_dir / "planner/topology_snapshot.json"
    state_path.write_text(
        json.dumps({"state": "FAILED", "overall_status": "FAIL_SOURCE"}) + "\n",
        encoding="utf-8",
    )
    run_summary_path.write_text(
        json.dumps({"completed_case_count": 90}) + "\n", encoding="utf-8"
    )
    case_summary_path.write_text(json.dumps({
        "status": "PASS_SINGLE_ROBOT_CASE_V2",
        "case": {
            "case_id": case_id,
            "world": "tunnel",
            "environment_seed": 23,
            "checkpoint_seed": 2,
            "method_family": "m1d_topology",
        },
        "planner_evidence": {
            "decision_trace": "planner/decision_trace.jsonl",
            "topology_snapshot": "planner/topology_snapshot.json",
        },
    }) + "\n", encoding="utf-8")
    nodes = [
        {"id": 0, "xyz_m": [0.0, 0.0, 0.0]},
        {"id": 1, "xyz_m": [8.0, 0.0, 0.0]},
    ]
    rows = [
        {"frame_index": 0, "route_arc_m": 0.0, "graph_update": {"node_id": 1}, "target": {"mode": "hold"}},
        {"frame_index": 1, "route_arc_m": 1.0, "graph_update": {"node_id": 1}, "target": {
            "mode": "graph_backtrack", "next_hop_node_id": 0,
            "graph_path_node_ids": [1, 0], "waypoint_xyz_m": [0.0, 0.0, 0.0],
            "frontier": {"node_id": 0, "stub_index": 0},
        }},
        {"frame_index": 2, "route_arc_m": 1.0, "graph_update": {"node_id": 1}, "target": {
            "mode": "graph_backtrack", "next_hop_node_id": 0,
            "graph_path_node_ids": [1, 0], "waypoint_xyz_m": [0.0, 0.0, 0.0],
            "frontier": {"node_id": 0, "stub_index": 0},
        }},
    ]
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    snapshot_path.write_text(
        json.dumps({"runtime": {"graph": {"nodes": nodes}}}) + "\n", encoding="utf-8"
    )
    expected = runner.audit_reanchor_evidence(rows, {"nodes": nodes})
    seal_path = source / "artifacts/evidence_sha256.txt"
    bound = (state_path, run_summary_path, case_summary_path, trace_path, snapshot_path)
    seal_path.write_text(
        "".join(
            f"{runner.sha256(path)}  {path.relative_to(tmp_path).as_posix()}\n" for path in bound
        ),
        encoding="utf-8",
    )
    spec = {
        "mechanism_source_run": "source",
        "mechanism_source_status": "FAIL_SOURCE",
        "mechanism_source_case_id": case_id,
        "mechanism_source_seal_sha256": runner.sha256(seal_path),
        "expected_source_mechanism": expected,
    }
    evidence = runner.validate_mechanism_source(spec, project_root=tmp_path)
    assert evidence["case_id"] == case_id
    assert evidence["mechanism_audit"] == expected
    assert evidence["uses_evaluator_gt"] is False


def test_probe_requires_sealed_readonly_compatibility_audit(tmp_path: Path) -> None:
    runner = _load()
    source = tmp_path / "compat"
    (source / "metrics").mkdir(parents=True)
    (source / "config").mkdir()
    (source / "artifacts").mkdir()
    paths = {
        "state": source / "RUN_STATE.json",
        "summary": source / "metrics/summary.json",
        "provenance": source / "config/source_provenance.json",
        "analysis": source / "metrics/stochastic_analysis.json",
    }
    paths["state"].write_text(
        json.dumps({"state": "COMPLETED", "overall_status": "PASS_COMPAT"}) + "\n",
        encoding="utf-8",
    )
    paths["summary"].write_text(json.dumps({
        "overall_status": "PASS_COMPAT",
        "source_case_count": 90,
        "source_mutation_count": 0,
        "raw_bag_reads": 0,
    }) + "\n", encoding="utf-8")
    paths["provenance"].write_text(json.dumps({
        "source_run": "source90",
        "source_seal_sha256": "a" * 64,
        "source_mutation_permitted": False,
    }) + "\n", encoding="utf-8")
    paths["analysis"].write_text(json.dumps({
        "compatibility_bridge": {
            "source_case_count": 90,
            "mapped_field": "status",
            "mapped_field_count": 90,
            "source_mutation_permitted": False,
        }
    }) + "\n", encoding="utf-8")
    seal_path = source / "artifacts/evidence_sha256.txt"
    seal_path.write_text(
        "".join(
            f"{runner.sha256(path)}  {path.relative_to(tmp_path).as_posix()}\n"
            for path in paths.values()
        ),
        encoding="utf-8",
    )
    spec = {
        "compatibility_audit_run": "compat",
        "compatibility_audit_status": "PASS_COMPAT",
        "compatibility_audit_seal_sha256": runner.sha256(seal_path),
        "mechanism_source_run": "source90",
        "mechanism_source_seal_sha256": "a" * 64,
    }
    evidence = runner.validate_compatibility_audit_source(spec, project_root=tmp_path)
    assert evidence["source_case_count"] == 90
    assert evidence["source_mutation_count"] == 0
