from __future__ import annotations

from copy import deepcopy

import pytest

from mtare_topo.evaluation.topology_reanchor_evidence import audit_reanchor_evidence


def snapshot():
    return {
        "nodes": [
            {"id": 0, "xyz_m": [0.0, 0.0, 0.75]},
            {"id": 1, "xyz_m": [8.0, 0.0, 0.75]},
        ]
    }


def row(frame, arc, *, mode="graph_backtrack", waypoint=None, current=1, next_hop=0):
    target = {
        "mode": mode,
        "frontier": {"node_id": 0, "stub_index": 1},
        "next_hop_node_id": next_hop,
        "graph_path_node_ids": [current, next_hop],
        "waypoint_xyz_m": [0.0, 0.0, 0.75] if waypoint is None else waypoint,
    }
    return {
        "frame_index": frame,
        "route_arc_m": arc,
        "graph_update": {"node_id": current},
        "target": target,
    }


def test_audit_quantifies_exact_verified_next_hop_stall() -> None:
    rows = [row(0, 0.0, mode="frontier_exit", current=0, next_hop=0)]
    rows += [row(index, float(index)) for index in range(1, 4)]
    rows += [row(index, 3.0) for index in range(4, 10)]
    result = audit_reanchor_evidence(rows, snapshot())
    assert result == {
        "schema_version": "topology_reanchor_evidence_v1",
        "frame_count": 10,
        "graph_backtrack_frame_count": 9,
        "exact_arrival_proxy_frame_count": 9,
        "exact_arrival_proxy_fraction": 0.9,
        "longest_constant_proxy_run_frames": 9,
        "first_proxy_frame": 1,
        "last_proxy_frame": 9,
        "proxy_frames_with_route_arc_growth": 3,
        "proxy_interval_route_arc_growth_m": 2.0,
        "last_route_arc_growth_frame": 3,
        "tail_without_route_arc_growth_frames": 6,
        "uses_evaluator_gt": False,
    }


def test_audit_requires_exact_node_waypoint_and_verified_path_prefix() -> None:
    wrong_waypoint = row(0, 1.0, waypoint=[0.0, 0.001, 0.75])
    wrong_path = row(1, 2.0); wrong_path["target"]["graph_path_node_ids"] = [1, 7, 0]
    result = audit_reanchor_evidence([wrong_waypoint, wrong_path], snapshot())
    assert result["graph_backtrack_frame_count"] == 2
    assert result["exact_arrival_proxy_frame_count"] == 0
    assert result["longest_constant_proxy_run_frames"] == 0


def test_evaluator_metadata_does_not_change_audit() -> None:
    rows = [row(0, 1.0)]
    alternate = deepcopy(rows)
    rows[0]["graph_update"]["evaluator_gt_edge_id"] = "secret_a"
    alternate[0]["graph_update"]["evaluator_gt_edge_id"] = "secret_b"
    assert audit_reanchor_evidence(rows, snapshot()) == audit_reanchor_evidence(alternate, snapshot())


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [row(1, 1.0)],
        [row(0, 2.0), row(1, 1.0)],
    ],
)
def test_invalid_trace_fails_closed(rows) -> None:
    with pytest.raises(ValueError):
        audit_reanchor_evidence(rows, snapshot())
