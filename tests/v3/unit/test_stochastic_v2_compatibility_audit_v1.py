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
        "run_stochastic_v2_compatibility_audit_v1",
        ROOT / "tools/v3/run_stochastic_v2_compatibility_audit_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scheduled() -> dict:
    return {
        "index": 0,
        "case_id": "000_tunnel_env11_m1d_seed0_repeat0",
        "block_id": "tunnel_env11",
        "method_family": "m1d_topology",
        "method_id": "m1d_seed0",
        "world": "tunnel",
        "environment_seed": 11,
        "execution_repeat": 0,
        "checkpoint_seed": 0,
        "runtime_sec": 600,
    }


def _summary() -> dict:
    scheduled = _scheduled()
    metrics = {metric: float(index + 1) for index, metric in enumerate(METRICS)}
    metrics.update(
        final_pose_xyz_m=[1.0, 2.0, 3.0],
        final_waypoint_xyz_m=[4.0, 5.0, 6.0],
        recording_audit={"passed": True},
    )
    return {
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


def test_case_integrity_audit_accepts_exact_schedule_and_evidence() -> None:
    runner = _load()
    runner._validate_case_summary(_summary(), _scheduled())


def test_original_mtare_does_not_require_replacement_planner_cycle_counter() -> None:
    runner = _load()
    scheduled = _scheduled()
    scheduled.update(method_family="original_mtare", method_id="original_mtare", checkpoint_seed=None)
    summary = _summary()
    summary["case"].update(scheduled)
    summary["planner_evidence"] = {"kind": "original_mtare_internal", "failed_cycles": None}
    summary["method_identity"] = {"method_family": "original_mtare"}
    runner._validate_case_summary(summary, scheduled)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["case"].__setitem__("world", "garage"), "schedule mismatch"),
        (lambda value: value["metrics"].__setitem__(METRICS[0], float("nan")), "non-finite metric"),
        (lambda value: value["metrics"]["recording_audit"].__setitem__("passed", False), "not PASS"),
        (lambda value: value["planner_evidence"].__setitem__("failed_cycles", 1), "planner evidence"),
        (lambda value: value["storage"].pop("archive_sha256"), "storage identity"),
    ],
)
def test_case_integrity_audit_fails_closed(mutation, message) -> None:
    runner = _load()
    summary = _summary()
    mutation(summary)
    with pytest.raises(RuntimeError, match=message):
        runner._validate_case_summary(summary, _scheduled())


def test_source_seal_parser_rejects_identity_drift(tmp_path: Path) -> None:
    runner = _load()
    source = tmp_path / "source"
    (source / "artifacts").mkdir(parents=True)
    seal = source / "artifacts/evidence_sha256.txt"
    seal.write_text("0" * 64 + "  source/RUN_STATE.json\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="seal identity drift"):
        runner._parse_seal(source, "f" * 64)


def test_execution_scope_rechecks_gate_authorization_and_tool_hashes(tmp_path: Path) -> None:
    runner = _load()
    tool = tmp_path / "tool.py"
    tool.write_text("pass\n", encoding="utf-8")
    spec = {
        "gate": 6,
        "operation": "audit",
        "user_authorization": {"status": "APPROVED"},
        "frozen_tools": {
            "runner": {"path": "tool.py", "sha256": runner.sha256(tool)},
        },
    }
    assert runner.validate_execution_scope(spec, project_root=tmp_path) == {
        "runner": runner.sha256(tool)
    }
    spec["operation"] = "closed_loop_single"
    with pytest.raises(RuntimeError, match="Gate-6 audit scope"):
        runner.validate_execution_scope(spec, project_root=tmp_path)


def test_load_source_verifies_exact_schedule_order_and_sealed_summaries(tmp_path: Path) -> None:
    runner = _load()
    runner.PROJECT_ROOT = tmp_path
    source = tmp_path / "source"
    (source / "metrics").mkdir(parents=True)
    (source / "config").mkdir()
    (source / "artifacts/cases").mkdir(parents=True)
    (source / "RUN_STATE.json").write_text(
        '{"state":"FAILED","overall_status":"FAIL_SOURCE"}\n', encoding="utf-8"
    )
    (source / "metrics/summary.json").write_text(
        '{"completed_case_count":90,"overall_status":"FAIL_SOURCE"}\n', encoding="utf-8"
    )

    schedule = []
    families = ("original_mtare", "m1d_topology", "layered_gt_map_oracle")
    index = 0
    for block in range(10):
        for family in families:
            for repeat in range(3):
                method_id = "m1d_seed" + str(repeat) if family == "m1d_topology" else family
                scheduled = {
                    "index": index,
                    "case_id": f"{index:03d}_case",
                    "block_id": f"block_{block}",
                    "method_family": family,
                    "method_id": method_id,
                    "world": "tunnel" if block < 5 else "garage",
                    "environment_seed": (11, 23, 37, 53, 71)[block % 5],
                    "execution_repeat": repeat,
                    "checkpoint_seed": repeat if family == "m1d_topology" else None,
                    "runtime_sec": 600,
                }
                summary = _summary()
                summary["case"].update(scheduled)
                summary["method_identity"] = {"method_family": family}
                summary["planner_evidence"] = {
                    "kind": "original_mtare_internal" if family == "original_mtare" else family,
                    "failed_cycles": None if family == "original_mtare" else 0,
                }
                case_dir = source / "artifacts/cases" / scheduled["case_id"]
                case_dir.mkdir()
                (case_dir / "summary.json").write_text(
                    json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
                )
                schedule.append(scheduled)
                index += 1
    schedule_path = source / "config/case_schedule.json"
    schedule_path.write_text(
        json.dumps({"cases": schedule}, sort_keys=True) + "\n", encoding="utf-8"
    )

    bound_paths = [
        source / "RUN_STATE.json",
        source / "metrics/summary.json",
        schedule_path,
        *sorted((source / "artifacts/cases").glob("*/summary.json")),
    ]
    seal_path = source / "artifacts/evidence_sha256.txt"
    seal_path.write_text(
        "".join(
            f"{runner.sha256(path)}  {path.relative_to(tmp_path).as_posix()}\n"
            for path in bound_paths
        ),
        encoding="utf-8",
    )
    spec = {
        "source_run": "source",
        "expected_source_state": "FAILED",
        "expected_source_status": "FAIL_SOURCE",
        "source_seal_sha256": runner.sha256(seal_path),
        "source_schedule_file_sha256": runner.sha256(schedule_path),
        "source_schedule_content_sha256": runner.schedule_content_sha256(schedule),
    }
    summaries, provenance, manifest = runner.load_source(spec)
    assert len(summaries) == 90
    assert manifest["case_count"] == 90
    assert [case["case_id"] for case in manifest["cases"]] == [case["case_id"] for case in schedule]
    assert manifest["schedule_file_sha256"] == runner.sha256(schedule_path)
    assert manifest["schedule_content_sha256"] == runner.schedule_content_sha256(schedule)
    assert provenance["verified_bound_file_count"] == 93


def test_mechanism_audit_reads_all_thirty_sealed_v9_traces_without_bags(
    tmp_path: Path,
) -> None:
    runner = _load()
    runner.PROJECT_ROOT = tmp_path
    source = tmp_path / "source"
    (source / "artifacts/cases").mkdir(parents=True)
    summaries = []
    bound = []
    for index in range(30):
        case_id = f"{index:03d}_m1d"
        case_dir = source / "artifacts/cases" / case_id
        (case_dir / "planner").mkdir(parents=True)
        trace_path = case_dir / "planner/decision_trace.jsonl"
        snapshot_path = case_dir / "planner/topology_snapshot.json"
        rows = [
            {
                "frame_index": 0,
                "route_arc_m": 0.0,
                "graph_update": {"node_id": 1},
                "target": {"mode": "hold"},
            },
            {
                "frame_index": 1,
                "route_arc_m": 1.0,
                "graph_update": {"node_id": 1},
                "target": (
                    {
                        "mode": "graph_backtrack",
                        "next_hop_node_id": 0,
                        "graph_path_node_ids": [1, 0],
                        "waypoint_xyz_m": [0.0, 0.0, 0.0],
                        "frontier": {"node_id": 0, "stub_index": 0},
                    }
                    if index == 0
                    else {"mode": "hold"}
                ),
            },
        ]
        trace_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        snapshot_path.write_text(
            json.dumps({
                "runtime": {
                    "graph": {
                        "nodes": [
                            {
                                "id": 0,
                                "xyz_m": [0.0, 0.0, 0.0],
                                "branch_count": 1,
                                "exit_stubs": [{"state": "observed"}],
                            },
                            {
                                "id": 1,
                                "xyz_m": [1.0, 0.0, 0.0],
                                "branch_count": 1,
                                "exit_stubs": [{"state": "traversed"}],
                            },
                        ]
                    }
                }
            }) + "\n",
            encoding="utf-8",
        )
        bound.extend((trace_path, snapshot_path))
        summaries.append({
            "case": {
                "case_id": case_id,
                "block_id": f"block_{index % 10}",
                "world": "tunnel",
                "environment_seed": index % 10,
                "checkpoint_seed": index % 3,
                "method_family": "m1d_topology",
            },
            "metrics": {
                "traveling_distance_m": float(index + 1),
                "final_explored_volume_m3": float(index + 2),
            },
            "method_identity": {"post_warmup_fallback_rate": 0.0},
            "planner_evidence": {
                "decision_trace": "planner/decision_trace.jsonl",
                "topology_snapshot": "planner/topology_snapshot.json",
            },
        })
    seal_path = source / "artifacts/evidence_sha256.txt"
    seal_path.write_text(
        "".join(
            f"{runner.sha256(path)}  {path.relative_to(tmp_path).as_posix()}\n"
            for path in bound
        ),
        encoding="utf-8",
    )
    result = runner.audit_source_reanchor_mechanisms(
        {
            "source_run": "source",
            "source_seal_sha256": runner.sha256(seal_path),
        },
        summaries,
    )
    assert result["case_count"] == 30
    assert result["case_count_with_exact_arrival_proxy"] == 1
    assert result["total_exact_arrival_proxy_frames"] == 1
    assert result["verified_trace_snapshot_file_count"] == 60
    assert result["maximum_constant_frontier_run_frames"] == 1
    assert result["maximum_node_exit_stub_minus_branch_count"] == 0
    assert result["raw_bag_reads"] == 0
    assert result["uses_evaluator_gt"] is False
