from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from mtare_topo.deployment.m1d_checkpoint import COMPOSITE_V9_MODE


ROOT = Path(__file__).resolve().parents[3]


def load(name: str, path: str):
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_v9_export_wrapper_freezes_new_identity() -> None:
    module = load("v9_export_wrapper", "tools/v3/run_aee_composite_v9_ros_deployment_export.py")
    assert module.RUN_ID == "gate6_20260822_aee_composite_v9_ros_deployment_export_seed20260822"
    assert module.SOURCE_STATUS == "PASS_AEE_CORRECTIVE_COMPOSITE_V9"


def test_v9_readiness_wrapper_freezes_new_identity() -> None:
    module = load("v9_readiness_wrapper", "tools/v3/run_aee_composite_v9_readiness.py")
    assert module.RUN_ID == "gate6_20260822_aee_composite_v9_readiness_v1r2_seed20260822"
    assert module.DEPLOYMENT_STATUS_PASS == "PASS_AEE_COMPOSITE_V9_ROS_DEPLOYMENT_EXPORT"


def test_v9_readiness_audit_requires_composite_trace(tmp_path: Path) -> None:
    runner = load("readiness_base_for_v9", "tools/v3/run_aee_adapted_readiness_v1.py")
    runner.EXPECTED_CHECKPOINT_MODE = COMPOSITE_V9_MODE
    case = {"case_id": "tunnel_seed0", "world": "tunnel", "environment_seed": 11,
            "checkpoint_seed": 0, "runtime_sec": 180.0}
    (tmp_path / "planner").mkdir()
    summary = {"status": "PASS_SINGLE_ROBOT_CASE_V2_PENDING_HOST_ARCHIVE",
               "case": {"case_id": "tunnel_seed0", "runtime_sec": 180.0},
               "metrics": {"recording_audit": {"passed": True}, "traveling_distance_m": 6.0},
               "planner_evidence": {"topology_snapshot": "planner/snapshot.json", "decision_trace": "planner/trace.jsonl"}}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    snapshot = {"failed_cycles": 0, "checkpoint": {"mode": COMPOSITE_V9_MODE},
                "post_warmup_cycles": 100, "post_warmup_fallback_cycles": 0,
                "runtime": {"graph": {"node_count": 2, "edges": [{"kind": "verified_traversed", "verified_traversal_count": 1}]}}}
    (tmp_path / "planner/snapshot.json").write_text(json.dumps(snapshot))
    good = {"target": {"mode": "frontier"}, "semantic_fallback": {"fallback_used": False},
            "composite_semantics": {"b0_branch_count": 2, "b0_role_index": 0, "neural_count_role_ignored": True}}
    (tmp_path / "planner/trace.jsonl").write_text(json.dumps(good) + "\n")
    assert runner.audit_case(tmp_path, case)["gates"]["post20_fallback_at_most_5pct"]
    del good["composite_semantics"]
    (tmp_path / "planner/trace.jsonl").write_text(json.dumps(good) + "\n")
    try:
        runner.audit_case(tmp_path, case)
    except RuntimeError as exc:
        assert "composite semantic audit" in str(exc)
    else:
        raise AssertionError("missing V9 composite audit must fail")
