from __future__ import annotations

from dataclasses import replace

import numpy as np

from mtare_topo.teacher.gse_causal_change_point_teacher import (
    CausalChangePointTeacherConfig,
    DirectionalGeometryProfile,
    assign_change_point_identity,
    causal_episode_matches_point,
    pair_bidirectional_change_points,
    past_only_causal_change_episodes,
    persistent_two_sided_change_episodes,
)


def _profile(
    width: np.ndarray,
    *,
    direction: int,
    edge_id: str = "edge_0",
    valid: np.ndarray | None = None,
) -> DirectionalGeometryProfile:
    values = np.asarray(width, dtype=np.float64)
    if direction == 1:
        values = values[::-1]
    length = float(len(values) - 1)
    return DirectionalGeometryProfile(
        traversal_id=f"{edge_id}:d{direction}",
        edge_id=edge_id,
        direction_index=direction,
        edge_length_m=length,
        traversal_arc_m=tuple(float(value) for value in np.arange(len(values))),
        width_m=tuple(float(value) for value in values),
        height_m=tuple(5.0 for _ in values),
        valid_mask=tuple(bool(value) for value in (np.ones(len(values), dtype=bool) if valid is None else valid)),
    )


def _paired_step():
    canonical_width = np.r_[np.full(20, 4.0), np.full(21, 8.0)]
    d0 = _profile(canonical_width, direction=0)
    d1 = _profile(canonical_width, direction=1)
    e0 = persistent_two_sided_change_episodes(d0)
    e1 = persistent_two_sided_change_episodes(d1)
    paired = pair_bidirectional_change_points(e0, e1)
    assert len(paired.accepted) == 1
    return d0, d1, paired.accepted[0]


def test_stable_step_is_bidirectional_and_direction_invariant() -> None:
    _, _, point = _paired_step()
    assert point.active_dimensions == ("width",)
    assert point.canonical_start_arc_m == 17.0
    assert point.canonical_end_arc_m == 22.0
    assert point.canonical_center_arc_m == 19.5
    assert point.direction0.canonical_width_delta_m > 0.0
    assert point.direction1.canonical_width_delta_m > 0.0


def test_short_spike_and_invalid_gap_do_not_become_persistent_nodes() -> None:
    spike = np.full(41, 4.0)
    spike[20:22] = 8.0
    assert persistent_two_sided_change_episodes(_profile(spike, direction=0)) == ()

    step = np.r_[np.full(20, 4.0), np.full(21, 8.0)]
    valid = np.ones(41, dtype=bool)
    valid[18] = False
    assert persistent_two_sided_change_episodes(_profile(step, direction=0, valid=valid)) == ()


def test_single_direction_and_ambiguous_matches_are_rejected() -> None:
    canonical_width = np.r_[np.full(20, 4.0), np.full(21, 8.0)]
    d0 = persistent_two_sided_change_episodes(_profile(canonical_width, direction=0))
    result = pair_bidirectional_change_points(d0, ())
    assert result.accepted == ()
    assert result.unilateral == d0

    d1 = persistent_two_sided_change_episodes(_profile(canonical_width, direction=1))
    duplicate = replace(d1[0], traversal_id="duplicate:d1")
    result = pair_bidirectional_change_points(d0, (d1[0], duplicate))
    assert result.accepted == ()
    assert len(result.ambiguous) == 3


def test_causal_detector_uses_closed_past_episode_and_backprojects_boundary() -> None:
    d0, d1, point = _paired_step()
    causal0 = past_only_causal_change_episodes(d0)
    causal1 = past_only_causal_change_episodes(d1)
    assert len(causal0) == len(causal1) == 1
    assert causal_episode_matches_point(causal0[0], point)
    assert causal_episode_matches_point(causal1[0], point)
    assert causal0[0].canonical_boundary_center_arc_m == 19.5
    assert causal1[0].canonical_boundary_center_arc_m == 19.5
    assert causal0[0].causal_delay_m > 0.0
    assert causal1[0].causal_delay_m > 0.0

    # A prefix ending at the recorded emission frame produces exactly the same
    # event, proving that samples after emission are not consulted.
    emission_index = int(causal0[0].emission_arc_m)
    prefix = DirectionalGeometryProfile(
        traversal_id=d0.traversal_id,
        edge_id=d0.edge_id,
        direction_index=0,
        edge_length_m=d0.edge_length_m,
        traversal_arc_m=d0.traversal_arc_m[: emission_index + 1],
        width_m=d0.width_m[: emission_index + 1],
        height_m=d0.height_m[: emission_index + 1],
        valid_mask=d0.valid_mask[: emission_index + 1],
    )
    assert past_only_causal_change_episodes(prefix) == causal0


def test_endpoint_identity_priority_and_interior_identity() -> None:
    _, _, point = _paired_step()
    degrees = {"n0": 2, "n1": 2}
    merged = assign_change_point_identity(
        parent_id="world",
        point=point,
        edge_length_m=40.0,
        from_node_id="n0",
        to_node_id="n1",
        node_degree_by_id=degrees,
        point_index=0,
    )
    assert merged.identity == "world:edge_0:geometry_change_point:000"
    assert merged.identity_kind == "interior_edge"

    near_start = replace(
        point,
        canonical_start_arc_m=4.0,
        canonical_end_arc_m=8.0,
        canonical_center_arc_m=6.0,
    )
    merged = assign_change_point_identity(
        parent_id="world",
        point=near_start,
        edge_length_m=40.0,
        from_node_id="n0",
        to_node_id="n1",
        node_degree_by_id=degrees,
        point_index=0,
    )
    assert merged.identity == "world:node:n0"
    assert merged.identity_kind == "degree_two_endpoint"

    suppressed = assign_change_point_identity(
        parent_id="world",
        point=near_start,
        edge_length_m=40.0,
        from_node_id="n0",
        to_node_id="n1",
        node_degree_by_id={"n0": 3, "n1": 2},
        point_index=0,
    )
    assert suppressed.identity is None
    assert suppressed.suppressing_node_id == "n0"


def test_short_edge_with_two_degree_two_endpoints_is_fail_closed() -> None:
    _, _, point = _paired_step()
    centered = replace(
        point,
        canonical_start_arc_m=6.0,
        canonical_end_arc_m=10.0,
        canonical_center_arc_m=8.0,
    )
    assignment = assign_change_point_identity(
        parent_id="world",
        point=centered,
        edge_length_m=16.0,
        from_node_id="n0",
        to_node_id="n1",
        node_degree_by_id={"n0": 2, "n1": 2},
        point_index=0,
        config=CausalChangePointTeacherConfig(endpoint_merge_radius_m=10.0),
    )
    assert assignment.identity is None
    assert assignment.identity_kind == "ambiguous_degree_two_endpoints"
