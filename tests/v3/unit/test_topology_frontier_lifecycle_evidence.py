from __future__ import annotations

import pytest

from mtare_topo.evaluation.topology_frontier_lifecycle_evidence import (
    audit_frontier_lifecycle_evidence,
)


def _graph() -> dict:
    return {
        "nodes": [
            {
                "id": 0,
                "branch_count": 2,
                "exit_stubs": [
                    {"state": "traversed"},
                    {"state": "observed"},
                    {"state": "observed"},
                ],
            },
            {
                "id": 1,
                "branch_count": 1,
                "exit_stubs": [{"state": "traversed"}],
            },
        ]
    }


def _row(frame: int, frontier: tuple[int, int] | None, *, mode: str = "frontier_exit") -> dict:
    target = {"mode": mode}
    if frontier is not None:
        target["frontier"] = {"node_id": frontier[0], "stub_index": frontier[1]}
    return {
        "frame_index": frame,
        "route_arc_m": float(frame),
        "graph_update": {"reason": "no_event", "node_id": 0},
        "target": target,
    }


def test_lifecycle_audit_reports_repeated_frontier_and_stub_surplus() -> None:
    rows = [
        _row(0, (0, 1)),
        _row(1, (0, 1)),
        _row(2, (0, 1)),
        _row(3, (1, 0)),
        _row(4, None, mode="hold"),
    ]
    result = audit_frontier_lifecycle_evidence(rows, _graph())
    assert result["frontier_target_frame_count"] == 4
    assert result["unique_frontier_target_count"] == 2
    assert result["most_selected_frontier"] == {
        "node_id": 0,
        "stub_index": 1,
        "frame_count": 3,
        "frame_fraction": 0.6,
    }
    assert result["longest_constant_frontier_run_frames"] == 3
    assert result["final_exit_stub_count"] == 4
    assert result["final_observed_exit_stub_count"] == 2
    assert result["maximum_node_exit_stub_minus_branch_count"] == 1
    assert result["uses_evaluator_gt"] is False


def test_lifecycle_audit_is_deterministic() -> None:
    rows = [_row(0, (0, 1)), _row(1, None, mode="graph_backtrack")]
    assert audit_frontier_lifecycle_evidence(rows, _graph()) == audit_frontier_lifecycle_evidence(
        rows, _graph()
    )


def test_lifecycle_audit_fails_on_invalid_frame_or_stub_state() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        audit_frontier_lifecycle_evidence([_row(1, (0, 1))], _graph())
    graph = _graph()
    graph["nodes"][0]["exit_stubs"][0]["state"] = "attempted"
    with pytest.raises(ValueError, match="invalid exit-stub state"):
        audit_frontier_lifecycle_evidence([_row(0, (0, 1))], graph)
