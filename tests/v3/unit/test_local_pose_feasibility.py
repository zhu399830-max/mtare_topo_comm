from mtare_topo.data.local_pose_feasibility import (
    contiguous_feasible_intervals,
    feasible_grid_components,
    compact_support_lateral_field,
    common_boolean_grid,
    nearest_common_feasible_offset,
    physical_groups,
)


def candidate(offset, feasible):
    return {"z_offset_m": offset, "feasible": feasible, "horizontal_clearance_m": 1.2, "downward_distance_m": 1.0, "upward_distance_m": 2.0}


def test_contiguous_intervals_and_nearest_zero_recommendation():
    records = [candidate(-.1, True), candidate(-.05, True), candidate(0, False), candidate(.05, True)]
    intervals = contiguous_feasible_intervals(records, .05)
    assert len(intervals) == 2
    assert intervals[0]["recommended_z_offset_m"] == -.05
    assert intervals[1]["minimum_z_offset_m"] == .05


def test_physical_groups_join_transitive_neighbours():
    records = [{"axis_xyz_m": [0, 0, 0]}, {"axis_xyz_m": [1, 0, 0]}, {"axis_xyz_m": [2, 0, 0]}, {"axis_xyz_m": [8, 0, 0]}]
    assert physical_groups(records, 1.1) == [[0, 1, 2], [3]]


def test_feasible_grid_components_use_four_neighbours_and_stable_order():
    records = [
        {"x_index": 0, "y_index": 0, "feasible": True},
        {"x_index": 1, "y_index": 0, "feasible": True},
        {"x_index": 2, "y_index": 1, "feasible": True},
        {"x_index": 2, "y_index": 2, "feasible": False},
        {"x_index": 5, "y_index": 5, "feasible": True},
    ]
    assert feasible_grid_components(records) == [[0, 1], [2], [4]]


def test_feasible_grid_components_empty_when_no_cell_is_feasible():
    assert feasible_grid_components([{"x_index": 0, "y_index": 0, "feasible": False}]) == []


def test_nearest_common_feasible_offset_is_shared_and_deterministic():
    candidates = {
        1: {(0.0, 0.0): False, (0.0, -.2): True, (0.0, .2): True},
        2: {(0.0, 0.0): False, (0.0, -.2): True, (0.0, .2): True},
    }
    assert nearest_common_feasible_offset(candidates, [1, 2]) == (0.0, -.2)


def test_compact_support_lateral_field_hits_anchor_and_is_zero_outside():
    import numpy as np
    values = compact_support_lateral_field(
        [[0, 0, 0], [10, 0, 0], [11, 0, 0]], [[0, 0, 0]], [[.2, -.1]], 10.0
    )
    np.testing.assert_allclose(values[0], [.2, -.1])
    np.testing.assert_allclose(values[1:], 0.0)


def test_common_boolean_grid_requires_every_observation():
    first = [
        {"x_index": 0, "y_index": 0, "x_offset_m": 0, "y_offset_m": 0, "feasible": True},
        {"x_index": 1, "y_index": 0, "x_offset_m": .1, "y_offset_m": 0, "feasible": True},
    ]
    second = [
        {"x_index": 0, "y_index": 0, "x_offset_m": 0, "y_offset_m": 0, "feasible": False},
        {"x_index": 1, "y_index": 0, "x_offset_m": .1, "y_offset_m": 0, "feasible": True},
    ]
    assert [item["feasible"] for item in common_boolean_grid([first, second])] == [False, True]
