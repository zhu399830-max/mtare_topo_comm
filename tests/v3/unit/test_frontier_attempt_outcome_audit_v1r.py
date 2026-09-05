from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_frontier_attempt_outcome_audit_v1r",
        ROOT / "tools/v3/run_frontier_attempt_outcome_audit_v1r.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event(frame, outcome, travel):
    return {
        "frame_index": frame,
        "outcome": outcome,
        "target_run_route_arc_m_before_event": travel,
    }


def _case(events):
    counts = {}
    for event in events:
        counts[event["outcome"]] = counts.get(event["outcome"], 0) + 1
    nonmatching = sum(
        outcome != "matched_verified_departure" for outcome in (
            event["outcome"] for event in events
        )
    )
    return {
        "waypoint_lookahead_m": 4.0,
        "minimum_event_travel_m": 8.0,
        "frontier_attempt": {
            "events": events,
            "outcome_counts": counts,
            "classified_attempt_event_count": len(events),
            "nonmatching_attempt_event_count": nonmatching,
        },
    }


def test_aggregate_recomputes_all_declared_counts():
    module = _load()
    first = _case([
        _event(1, "matched_verified_departure", 8.0),
        _event(2, "divergent_verified_departure", 5.0),
        _event(3, "same_node_loop_merge", 9.0),
    ])
    value = module.summarize_frontier_attempt_cases_v1r(
        [first, *[_case([]) for _ in range(29)]]
    )
    assert value["case_count_with_nonmatching_attempt"] == 1
    assert value["nonmatching_attempt_event_count"] == 2
    assert value["nonmatching_at_least_frozen_waypoint_lookahead_count"] == 2
    assert value["nonmatching_at_least_frozen_minimum_event_travel_count"] == 1
    assert value["uses_new_tuned_threshold"] is False


@pytest.mark.parametrize("field", [
    "outcome_counts", "classified_attempt_event_count", "nonmatching_attempt_event_count"
])
def test_aggregate_rejects_internal_count_drift(field):
    module = _load()
    cases = [_case([_event(1, "divergent_verified_departure", 8.0)])]
    cases.extend(_case([]) for _ in range(29))
    value = cases[0]["frontier_attempt"][field]
    cases[0]["frontier_attempt"][field] = {} if isinstance(value, dict) else value + 1
    with pytest.raises(RuntimeError, match="count"):
        module.summarize_frontier_attempt_cases_v1r(cases)


def _material_run(module, root: Path):
    run_id = "gate6_20260823_frontier_attempt_test_seed0"
    spec = {
        "run_id": run_id,
        "config_path": "configs/spec.json",
        "data_card": "configs/card.json",
        "command": ["python", "runner.py", "--run", run_id],
    }
    (root / "configs").mkdir()
    spec_path = root / spec["config_path"]
    card_path = root / spec["data_card"]
    spec_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    card_path.write_text(json.dumps({"card": "exact"}) + "\n", encoding="utf-8")
    run_dir = root / "results/gate6_single_robot" / run_id
    for child in ("config", "logs", "metrics", "previews", "artifacts"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    (run_dir / "config/source_config").write_bytes(spec_path.read_bytes())
    for name, value in (
        ("run_spec.json", spec),
        ("data_card.json", {"card": "exact"}),
        ("status_snapshot.json", {}),
        ("environment.json", {}),
    ):
        (run_dir / "config" / name).write_text(json.dumps(value) + "\n", encoding="utf-8")
    (run_dir / "config/command.txt").write_text(
        shlex.join(spec["command"]) + "\n", encoding="utf-8"
    )
    (run_dir / "RUN_STATE.json").write_text(json.dumps({
        "run_id": run_id, "state": "CREATED_NOT_EXECUTED"
    }) + "\n", encoding="utf-8")
    return spec_path, spec, run_dir


def test_material_run_identity_requires_exact_create_run_snapshots(tmp_path):
    module = _load()
    spec_path, spec, run_dir = _material_run(module, tmp_path)
    module.validate_material_run_identity(
        spec_path, spec, run_dir, project_root=tmp_path
    )
    (run_dir / "config/command.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="command snapshot drift"):
        module.validate_material_run_identity(
            spec_path, spec, run_dir, project_root=tmp_path
        )


def test_append_only_source_identity_binds_source_and_local_hash(tmp_path):
    module = _load()
    graph = tmp_path / "graph.py"
    graph.write_text("# append-only identity\n", encoding="utf-8")
    identity = {"path": "graph.py", "sha256": module.sha256(graph)}
    source_config = tmp_path / "source/config"
    source_config.mkdir(parents=True)
    (source_config / "run_spec.json").write_text(json.dumps({
        "frozen_tools": {"causal_graph": identity}
    }) + "\n", encoding="utf-8")
    spec = {
        "source_run": "source",
        "source_causal_graph_path": identity["path"],
        "source_causal_graph_sha256": identity["sha256"],
    }
    assert module.validate_append_only_source_identity(
        spec, project_root=tmp_path
    ) == identity
    spec["source_causal_graph_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="identity drift"):
        module.validate_append_only_source_identity(spec, project_root=tmp_path)
