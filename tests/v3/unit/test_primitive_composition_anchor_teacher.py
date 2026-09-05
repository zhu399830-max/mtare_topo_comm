import copy

import numpy as np
import pytest

from mtare_topo.evaluation.primitive_composition_anchor_teacher import (
    construction_endpoint_anchor_table,
    current_sensor_anchor_targets,
    observable_anchor_pair_slice,
)


def _document():
    primitives = []
    operations = []
    members_by_node = {"n0": [], "n1": [], "n2": [], "n3": []}
    anchors = {
        "n0": [0.0, 0.0, 0.0], "n1": [1.0, 0.0, 0.0],
        "n2": [0.0, 1.0, 0.0], "n3": [2.0, 0.0, 1.0],
    }
    endpoint_nodes = (("n0", "n1"), ("n0", "n2"), ("n0", "n3"))
    for primitive_index, nodes in enumerate(endpoint_nodes):
        primitive_id = f"p{primitive_index}"
        endpoints = []
        for endpoint_index, node_id in enumerate(nodes):
            value = {
                "primitive_id": primitive_id, "endpoint_index": endpoint_index,
                "node_id": node_id, "xyz_m": anchors[node_id],
                "composition_anchor_xyz_m": anchors[node_id],
            }
            endpoints.append(value); members_by_node[node_id].append(copy.deepcopy(value))
        primitives.append({"primitive_id": primitive_id, "endpoints": endpoints})
    for node_id, members in members_by_node.items():
        operations.append({
            "node_id": node_id, "anchor_xyz_m": anchors[node_id],
            "degree": len(members), "member_endpoints": members,
        })
    return {
        "base_construction": {
            "primitives": primitives, "composition_operations": operations,
        },
        "realized_primitives": [{"primitive_id": f"p{i}"} for i in range(3)],
    }


def test_construction_anchor_table_proves_exactly_one_membership():
    result = construction_endpoint_anchor_table(_document())
    assert result.primitive_ids == ("p0", "p1", "p2")
    assert result.node_ids[0, 0] == "n0"
    np.testing.assert_array_equal(result.anchor_world_m[:3, 0], np.zeros((3, 3)))


def test_duplicate_or_missing_composition_membership_fails():
    document = _document()
    document["base_construction"]["composition_operations"][0]["member_endpoints"].append(
        copy.deepcopy(document["base_construction"]["composition_operations"][0]["member_endpoints"][0])
    )
    document["base_construction"]["composition_operations"][0]["degree"] += 1
    with pytest.raises(ValueError, match="duplicate"):
        construction_endpoint_anchor_table(document)


def test_anchor_targets_are_in_current_sensor_frame():
    table = construction_endpoint_anchor_table(_document())
    index = np.full((1, 32), -1, dtype=np.int32); index[0, :3] = (0, 1, 2)
    mask = (index >= 0).astype(np.uint8)
    target = current_sensor_anchor_targets(
        primitive_index=index, primitive_mask=mask,
        anchor_world_m=table.anchor_world_m,
        sensor_xyz_m=np.asarray([[1.0, 0.0, 0.0]]),
        yaw_deg=np.asarray([90.0]),
    )
    np.testing.assert_allclose(target[0, :3, 0], [[0.0, 1.0, 0.0]] * 3, atol=1e-12)
    assert np.isnan(target[0, 3:]).all()


def test_observable_pair_slice_separates_shared_anchor_and_hard_negative():
    table = construction_endpoint_anchor_table(_document())
    index = np.full((1, 32), -1, dtype=np.int32); index[0, :3] = (0, 1, 2)
    mask = (index >= 0).astype(np.uint8)
    anchors = current_sensor_anchor_targets(
        primitive_index=index, primitive_mask=mask,
        anchor_world_m=table.anchor_world_m,
        sensor_xyz_m=np.zeros((1, 3)), yaw_deg=np.zeros(1),
    )
    axis = np.zeros((1, 32, 3, 3), dtype=np.float32)
    for slot in range(3):
        axis[0, slot, 0] = anchors[0, slot, 0]
        axis[0, slot, 2] = anchors[0, slot, 1]
    observed = np.zeros((1, 32, 2), dtype=np.uint8)
    observed[0, :3] = 1
    neighbor = np.full((1, 32, 2, 3), -1, dtype=np.int8)
    for source in (0, 2, 4):
        values = sorted({0, 2, 4} - {source})
        neighbor[0, source // 2, source % 2, :2] = values
    overlap = np.zeros((1, 32, 32), dtype=np.uint8)
    overlap[0, 0, 1] = overlap[0, 1, 0] = 1
    result = observable_anchor_pair_slice(
        axis_control_current_sensor_m=axis,
        anchor_current_sensor_m=anchors,
        primitive_mask=mask, endpoint_observed=observed,
        endpoint_neighbor=neighbor, disconnected_overlap=overlap,
    )
    assert len(result.attachment_target) == 12
    assert int(result.attachment_target.sum()) == 3
    assert np.max(result.target_anchor_pair_distance_m[result.attachment_target]) == 0.0
    assert np.min(result.target_anchor_pair_distance_m[~result.attachment_target]) > 0.0
    assert np.any(result.overlap_hard_negative)


def test_observed_inactive_endpoint_fails_closed():
    index = np.full((1, 32), -1, dtype=np.int32); index[0, 0] = 0
    mask = (index >= 0).astype(np.uint8)
    anchors = np.full((1, 32, 2, 3), np.nan); anchors[0, 0] = 0
    axis = np.zeros((1, 32, 3, 3))
    observed = np.zeros((1, 32, 2), dtype=np.uint8); observed[0, 1, 0] = 1
    neighbor = np.full((1, 32, 2, 3), -1, dtype=np.int8)
    overlap = np.zeros((1, 32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="inactive"):
        observable_anchor_pair_slice(
            axis_control_current_sensor_m=axis,
            anchor_current_sensor_m=anchors,
            primitive_mask=mask, endpoint_observed=observed,
            endpoint_neighbor=neighbor, disconnected_overlap=overlap,
        )
