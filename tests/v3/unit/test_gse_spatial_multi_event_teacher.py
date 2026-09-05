from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.teacher.gse_spatial_multi_event_teacher import (
    event_type_for_degree,
    has_opposite_headings,
    robot_relative_xyz,
    spatial_structure_event_targets,
)


def _graph(*nodes):
    return {"nodes": list(nodes), "edges": []}


def _node(node_id, degree, xyz):
    return {"id": node_id, "degree": degree, "xyz": list(xyz)}


def _clear_raycast(origins, directions):
    return np.full((origins.shape[0], 1), np.inf, dtype=np.float64)


def test_degree_mapping_and_non_event_degree():
    assert event_type_for_degree(1) == "terminal"
    assert event_type_for_degree(3) == "junction"
    assert event_type_for_degree(8) == "junction"
    assert event_type_for_degree(2) is None


def test_robot_yaw_coordinates_are_forward_left_up():
    relative = robot_relative_xyz((0.0, 0.0, 0.0), 90.0, (1.0, 0.0, 2.0))
    assert relative == pytest.approx((0.0, -1.0, 2.0))


def test_opposite_heading_coverage_uses_a_frozen_120_degree_separation():
    assert has_opposite_headings([5.0, 125.0]) is True
    assert has_opposite_headings([350.0, 110.0]) is True
    assert has_opposite_headings([5.0, 124.9]) is False
    assert has_opposite_headings([20.0]) is False
    assert has_opposite_headings([0.0, float("nan")]) is False


def test_same_frame_returns_terminal_and_junction_targets():
    graph = _graph(_node("terminal", 1, (4.0, 0.0, 0.0)),
                   _node("junction", 3, (5.0, 0.0, 0.0)),
                   _node("corridor", 2, (1.0, 0.0, 0.0)))
    targets = spatial_structure_event_targets(
        parent_id="world:frame:7", graph=graph,
        current_sensor_xyz_m=(0.0, 0.0, 0.0), robot_yaw_deg=0.0,
        fta_distance_m=0.0, cast_distances=_clear_raycast,
    )
    assert {target.event_type for target in targets} == {"terminal", "junction"}
    assert {target.node_id for target in targets} == {"terminal", "junction"}


def test_occluded_target_is_marked_invisible():
    graph = _graph(_node("junction", 3, (10.0, 0.0, 0.0)))

    def blocked(origins, directions):
        return np.full((origins.shape[0], 1), 2.0, dtype=np.float64)

    target = spatial_structure_event_targets(
        parent_id="world:frame:8", graph=graph,
        current_sensor_xyz_m=(0.0, 0.0, 0.0), robot_yaw_deg=0.0,
        fta_distance_m=0.0, cast_distances=blocked,
    )[0]
    assert target.visible is False
    assert target.first_hit_distance_m == pytest.approx(2.0)


def test_nonfinite_raycast_distance_is_an_unoccluded_ray():
    graph = _graph(_node("terminal", 1, (10.0, 0.0, 0.0)))
    target = spatial_structure_event_targets(
        parent_id="world:frame:9", graph=graph,
        current_sensor_xyz_m=(0.0, 0.0, 0.0), robot_yaw_deg=0.0,
        fta_distance_m=0.0, cast_distances=_clear_raycast,
    )[0]
    assert target.visible is True
    assert target.first_hit_distance_m is None


def test_targets_beyond_fifty_meters_are_filtered():
    graph = _graph(_node("near", 1, (49.0, 0.0, 0.0)),
                   _node("far", 3, (50.0, 0.0, 0.0)))
    targets = spatial_structure_event_targets(
        parent_id="world:frame:10", graph=graph,
        current_sensor_xyz_m=(0.0, 0.0, 0.0), robot_yaw_deg=0.0,
        fta_distance_m=0.0, cast_distances=_clear_raycast,
    )
    assert [target.node_id for target in targets] == ["near"]


def test_duplicate_nodes_and_wrong_raycast_shape_are_rejected():
    duplicate = _graph(_node("same", 1, (1.0, 0.0, 0.0)),
                       _node("same", 3, (2.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="node identity"):
        spatial_structure_event_targets(
            parent_id="world:frame:11", graph=duplicate,
            current_sensor_xyz_m=(0.0, 0.0, 0.0), robot_yaw_deg=0.0,
            fta_distance_m=0.0, cast_distances=_clear_raycast,
        )

    graph = _graph(_node("terminal", 1, (1.0, 0.0, 0.0)))

    def malformed(origins, directions):
        return np.zeros(origins.shape[0], dtype=np.float64)

    with pytest.raises(ValueError, match="raycast result"):
        spatial_structure_event_targets(
            parent_id="world:frame:12", graph=graph,
            current_sensor_xyz_m=(0.0, 0.0, 0.0), robot_yaw_deg=0.0,
            fta_distance_m=0.0, cast_distances=malformed,
        )
