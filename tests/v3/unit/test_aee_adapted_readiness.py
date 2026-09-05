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
        "run_aee_adapted_readiness_v1", ROOT / "tools/v3/run_aee_adapted_readiness_v1.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _schedule():
    return [
        {"case_id": f"{world}_seed{seed}", "world": world, "environment_seed": 11, "checkpoint_seed": seed, "runtime_sec": 180.0}
        for world in ("tunnel", "garage") for seed in (0, 1, 2)
    ]


def test_schedule_is_exact_two_world_by_three_checkpoint_grid() -> None:
    runner = _load()
    assert len(runner.validate_schedule(_schedule())) == 6
    with pytest.raises(RuntimeError):
        runner.validate_schedule(_schedule()[:-1])
    bad = _schedule()
    bad[-1] = {**bad[-1], "checkpoint_seed": 1}
    with pytest.raises(RuntimeError):
        runner.validate_schedule(bad)


def test_case_command_freezes_180s_host_archive_and_adapted_checkpoint(tmp_path: Path) -> None:
    runner = _load()
    case = _schedule()[0]
    command, name = runner.case_command(tmp_path, case, {"seed": 0, "path": "adapted.pt", "sha256": "abc"})
    shell = command[-1]
    assert name == "aee-readiness-tunnel_seed0"
    assert command[command.index("--hostname") + 1] == "localhost"
    for token in ("--runtime-sec 180.0", "--archive-mode host", "--host-uid 1000", "--host-gid 1000", "--checkpoint adapted.pt", "--checkpoint-sha256 abc"):
        assert token in shell
    assert "--topic-contract /workspace/configs/v3/gate5/closed_loop_recording_topics_v1.json" in shell


def test_case_command_can_freeze_aee_raw_recording_contract(tmp_path: Path) -> None:
    runner = _load()
    command, _ = runner.case_command(
        tmp_path,
        _schedule()[0],
        {"seed": 0, "path": "adapted.pt", "sha256": "abc"},
        "configs/v3/gate6/closed_loop_recording_topics_aee_v2.json",
    )
    assert "--topic-contract /workspace/configs/v3/gate6/closed_loop_recording_topics_aee_v2.json" in command[-1]


def _case_tree(root: Path, fallback: int = 4) -> None:
    (root / "evidence").mkdir(parents=True)
    (root / "planner").mkdir()
    summary = {
        "status": "PASS_SINGLE_ROBOT_CASE_V2_PENDING_HOST_ARCHIVE",
        "case": {"case_id": "tunnel_seed0", "runtime_sec": 180.0},
        "metrics": {"recording_audit": {"passed": True}, "traveling_distance_m": 7.5},
        "planner_evidence": {"topology_snapshot": "planner/topology_snapshot.json", "decision_trace": "planner/decision_trace.jsonl"},
    }
    (root / "summary.json").write_text(json.dumps(summary))
    snapshot = {
        "failed_cycles": 0,
        "checkpoint": {"mode": "M1D_AEE_HEAD_ADAPTED_V1"},
        "post_warmup_cycles": 100,
        "post_warmup_fallback_cycles": fallback,
        "runtime": {"graph": {"node_count": 2, "edges": [{"kind": "verified_traversed", "verified_traversal_count": 1}]}},
    }
    (root / "planner/topology_snapshot.json").write_text(json.dumps(snapshot))
    records = [
        {"target": {"mode": "frontier" if index == 0 else "hold"}, "semantic_fallback": {"fallback_used": index < fallback}}
        for index in range(120)
    ]
    (root / "planner/decision_trace.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records))


def test_case_audit_enforces_movement_graph_waypoint_and_fallback(tmp_path: Path) -> None:
    runner = _load()
    case_dir = tmp_path / "case"
    _case_tree(case_dir, fallback=4)
    result = runner.audit_case(case_dir, _schedule()[0])
    assert result["post_warmup_fallback_rate"] == 0.04
    assert all(result["gates"].values())
    snapshot = json.loads((case_dir / "planner/topology_snapshot.json").read_text())
    snapshot["post_warmup_fallback_cycles"] = 6
    (case_dir / "planner/topology_snapshot.json").write_text(json.dumps(snapshot))
    with pytest.raises(RuntimeError, match="scientific gate failed"):
        runner.audit_case(case_dir, _schedule()[0])
