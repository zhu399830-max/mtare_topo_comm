"""Synthetic-only contracts: visible fragments are not node visibility labels."""

import copy

import numpy as np
import pytest

from mtare_topo.data.gse_local_teacher_audit import (
    audit_support,
    decode_and_validate_relations,
    expected_construction_attachment,
    score_cached_geometry,
)


def _relations():
    mask = np.zeros((2, 32), dtype=np.uint8)
    mask[:, :3] = 1
    neighbors = np.full((2, 32, 2, 3), -1, dtype=np.int8)
    neighbors[0, 0, 1, 0] = 2  # primitive 0 endpoint 1 <-> primitive 1 endpoint 0.
    neighbors[0, 1, 0, 0] = 1
    overlap = np.zeros((2, 32, 32), dtype=np.uint8)
    overlap[0, 0, 2] = overlap[0, 2, 0] = 1
    return neighbors, np.packbits(overlap, axis=-1, bitorder="little"), mask


def test_relations_decode_orientation_little_endian_and_batch_isolation():
    neighbors, packed, mask = _relations()
    attachment, overlap = decode_and_validate_relations(neighbors, packed, mask)
    assert attachment.shape == (2, 32, 2, 32, 2)
    assert overlap.shape == (2, 32, 32)
    assert attachment[0, 0, 1, 1, 0] == attachment[0, 1, 0, 0, 1] == 1
    assert attachment.sum() == 2
    assert overlap[0, 0, 2] == overlap[0, 2, 0] == 1
    assert overlap.sum() == 2
    assert not attachment[1].any() and not overlap[1].any()


@pytest.mark.parametrize("case", [
    "below_sentinel", "out_of_range", "duplicate", "asymmetric_attachment",
    "same_endpoint", "same_primitive", "invalid_source", "invalid_destination",
    "nonbinary_mask", "fractional_neighbor", "wrong_neighbor_shape",
])
def test_invalid_attachment_records_fail_closed(case):
    neighbors, packed, mask = _relations()
    if case == "below_sentinel":
        neighbors[0, 0, 0, 0] = -2
    elif case == "out_of_range":
        neighbors[0, 0, 0, 0] = 64
    elif case == "duplicate":
        neighbors[0, 0, 1, 1] = 2
    elif case == "asymmetric_attachment":
        neighbors[0, 1, 0, 0] = -1
    elif case == "same_endpoint":
        neighbors[0, 0, 0, 0] = 0
    elif case == "same_primitive":
        neighbors[0, 0, 0, 0] = 1
        neighbors[0, 0, 1, 1] = 0
    elif case == "invalid_source":
        neighbors[0, 3, 0, 0] = 0
    elif case == "invalid_destination":
        neighbors[0, 0, 0, 0] = 6
    elif case == "nonbinary_mask":
        mask[0, 2] = 2
    elif case == "fractional_neighbor":
        neighbors = neighbors.astype(np.float64)
        neighbors[0, 0, 0, 0] = 1.5
    else:
        neighbors = neighbors[..., :2]
    with pytest.raises(ValueError):
        decode_and_validate_relations(neighbors, packed, mask)


@pytest.mark.parametrize("case", ["asymmetric", "self", "invalid_slot", "attached_pair"])
def test_angular_overlap_validation_never_interprets_it_as_physical_overlap(case):
    neighbors, packed, mask = _relations()
    overlap = np.unpackbits(packed, axis=-1, count=32, bitorder="little")
    if case == "asymmetric":
        overlap[0, 2, 0] = 0
    elif case == "self":
        overlap[0, 0, 0] = 1
    elif case == "invalid_slot":
        overlap[0, 0, 3] = overlap[0, 3, 0] = 1
    else:
        overlap[0, 0, 1] = overlap[0, 1, 0] = 1
    with pytest.raises(ValueError):
        decode_and_validate_relations(neighbors, np.packbits(overlap, axis=-1, bitorder="little"), mask)


def _support():
    mask = np.zeros((2, 32), dtype=np.uint8)
    mask[:, :2] = 1
    counts = np.zeros((2, 32), dtype=np.int32)
    counts[:, :2] = (100, 2)
    temporal = np.zeros((2, 5, 32), dtype=np.uint8)
    temporal[:, :, 0] = 1
    temporal[:, :2, 1] = 1
    return mask, counts, temporal


def test_support_checks_presence_not_full_visibility_and_accepts_empty_rows():
    mask, counts, temporal = _support()
    assert audit_support(mask, counts, temporal) is None
    assert audit_support(np.zeros_like(mask), np.zeros_like(counts), np.zeros_like(temporal)) is None


@pytest.mark.parametrize("case", [
    "negative", "fractional", "nan", "nonbinary_mask", "nonbinary_temporal",
    "count_without_mask", "mask_without_count", "temporal_without_mask",
    "count_without_temporal", "fewer_rays_than_visible_frames", "shape",
])
def test_inconsistent_support_fails(case):
    mask, counts, temporal = _support()
    if case == "negative":
        counts[0, 1] = -1
    elif case in ("fractional", "nan"):
        counts = counts.astype(np.float64)
        counts[0, 1] = 1.5 if case == "fractional" else np.nan
    elif case == "nonbinary_mask":
        mask[0, 1] = 2
    elif case == "nonbinary_temporal":
        temporal[0, 0, 1] = 2
    elif case == "count_without_mask":
        mask[0, 1] = 0
    elif case == "mask_without_count":
        counts[0, 1] = 0
    elif case == "temporal_without_mask":
        temporal[0, 0, 2] = 1
    elif case == "count_without_temporal":
        temporal[0, :, 1] = 0
    elif case == "fewer_rays_than_visible_frames":
        counts[0, 1] = 1
    else:
        temporal = temporal[:, :4]
    with pytest.raises(ValueError):
        audit_support(mask, counts, temporal)


def _construction():
    # Two distinct edge-level primitives intentionally share a physical tunnel.
    return {
        "realized_primitives": [
            {"primitive_id": "edge_a", "source_tunnel_id": "through"},
            {"primitive_id": "edge_b", "source_tunnel_id": "through"},
            {"primitive_id": "edge_c", "source_tunnel_id": "branch"},
        ],
        "base_construction": {"composition_operations": [
            {"node_id": "node_left", "degree": 2, "member_endpoints": [
                {"primitive_id": "edge_a", "endpoint_index": 1},
                {"primitive_id": "edge_b", "endpoint_index": 0},
            ]},
            {"node_id": "node_right", "degree": 2, "member_endpoints": [
                {"primitive_id": "edge_b", "endpoint_index": 1},
                {"primitive_id": "edge_c", "endpoint_index": 0},
            ]},
        ]},
    }


def test_construction_retains_multiple_nodes_and_same_tunnel_edge_incidence():
    indices = np.full((2, 32), -1, dtype=np.int32)
    indices[0, :3] = (2, 0, 1)  # Storage order is deliberately not primitive order.
    indices[1, :2] = (0, 1)  # The invisible third edge never gets a phantom slot.
    mask = (indices >= 0).astype(np.uint8)
    result = expected_construction_attachment(indices, mask, _construction())
    assert result.shape == (2, 32, 2, 32, 2)
    assert result[0, 1, 1, 2, 0] == result[0, 2, 0, 1, 1] == 1
    assert result[0, 2, 1, 0, 0] == result[0, 0, 0, 2, 1] == 1
    assert result[0].sum() == 4
    assert result[1, 0, 1, 1, 0] == result[1, 1, 0, 0, 1] == 1
    assert result[1].sum() == 2


def test_construction_order_and_arbitrary_node_ids_do_not_change_attachment():
    indices = np.full((1, 32), -1, dtype=np.int32)
    indices[0, :3] = (0, 1, 2)
    mask = (indices >= 0).astype(np.uint8)
    original = _construction()
    changed = copy.deepcopy(original)
    changed["base_construction"]["composition_operations"].reverse()
    for composition in changed["base_construction"]["composition_operations"]:
        composition["node_id"] += "_renamed"
        composition["member_endpoints"].reverse()
    np.testing.assert_array_equal(
        expected_construction_attachment(indices, mask, original),
        expected_construction_attachment(indices, mask, changed),
    )


def _cached_geometry(active_count=2):
    teacher = {
        "primitive_mask": np.zeros((1, 32), dtype=np.uint8),
        "axis_control_current_sensor_m": np.zeros((1, 32, 3, 3), dtype=np.float32),
        "endpoint_half_axes_m": np.ones((1, 32, 2, 2), dtype=np.float32),
        "endpoint_shape_exponent": np.ones((1, 32, 2), dtype=np.float32),
        "temporal_visibility": np.zeros((1, 5, 32), dtype=np.uint8),
        "endpoint_attachment": np.zeros((1, 32, 2, 32, 2), dtype=np.uint8),
        "angular_overlap": np.zeros((1, 32, 32), dtype=np.uint8),
        "support_ray_count": np.zeros((1, 32), dtype=np.int32),
    }
    for slot in range(active_count):
        teacher["primitive_mask"][0, slot] = 1
        teacher["axis_control_current_sensor_m"][0, slot] = (
            (0, 10 * slot, 0), (2, 10 * slot, 0), (5, 10 * slot, 0))
        teacher["endpoint_half_axes_m"][0, slot] = ((1 + slot, 2), (3 + slot, 4))
        teacher["endpoint_shape_exponent"][0, slot] = (2, 4 + slot)
        teacher["temporal_visibility"][0, :slot + 1, slot] = 1
        teacher["support_ray_count"][0, slot] = 11 + slot
    prediction = {
        "axis_control_current_sensor_m": np.full((1, 32, 3, 3), 100, dtype=np.float32),
        "endpoint_half_axes_m": np.ones((1, 32, 2, 2), dtype=np.float32),
        "endpoint_shape_exponent": np.ones((1, 32, 2), dtype=np.float32),
        "existence_logits": np.zeros((1, 32), dtype=np.float32),
        "geometry_uncertainty": np.full((1, 32), .7, dtype=np.float32),
        "endpoint_evidence_logits": np.full((1, 32, 2), -100, dtype=np.float32),
    }
    return prediction, teacher


@pytest.mark.parametrize("reverse", [False, True])
def test_cached_geometry_exact_matches_are_zero_error_under_endpoint_reversal(reverse):
    prediction, teacher = _cached_geometry()
    for predicted, truth in ((5, 0), (17, 1)):
        for key in ("axis_control_current_sensor_m", "endpoint_half_axes_m", "endpoint_shape_exponent"):
            values = teacher[key][0, truth]
            prediction[key][0, predicted] = values[::-1] if reverse else values
    rows = score_cached_geometry(prediction, teacher)
    assert len(rows) == 1 and len(rows[0]) == 2
    for record, (predicted, truth) in zip(rows[0], ((5, 0), (17, 1))):
        assert record["prediction_slot"] == predicted
        assert record["teacher_slot_scoring_only"] == truth
        assert record["reversed"] is reverse
        for key in ("axis_coordinate_mae_m", "axis_control_point_mean_euclidean_m",
                    "half_axes_mae_m", "shape_exponent_mae"):
            assert record[key] == 0
        assert record["existence_probability"] == .5
        assert record["teacher_support_rays_nonexclusive"] == 11 + truth
        assert record["teacher_support_frames"] == truth + 1


def test_cached_geometry_assignment_is_not_detection_even_with_low_existence_and_large_error():
    prediction, teacher = _cached_geometry(active_count=1)
    for key in ("axis_control_current_sensor_m", "endpoint_half_axes_m", "endpoint_shape_exponent"):
        prediction[key][0, 5] = teacher[key][0, 0]
    prediction["axis_control_current_sensor_m"][0, 5, :, 0] += 3
    prediction["existence_logits"][0, 5] = -100
    records = score_cached_geometry(prediction, teacher)[0]
    assert len(records) == 1
    record = records[0]
    assert record["prediction_slot"] == 5
    assert record["existence_probability"] < 1e-30
    assert record["axis_coordinate_mae_m"] == 1
    assert record["axis_control_point_mean_euclidean_m"] == 3
    assert set(record) == {
        "prediction_slot", "teacher_slot_scoring_only", "reversed",
        "axis_coordinate_mae_m", "axis_control_point_mean_euclidean_m",
        "half_axes_mae_m", "shape_exponent_mae", "existence_probability",
        "geometry_uncertainty_uncalibrated", "teacher_support_rays_nonexclusive",
        "teacher_support_frames",
    }  # No detected/accepted/visible-node decision is licensed by assignment.


@pytest.mark.parametrize("field", [
    "axis_control_current_sensor_m", "endpoint_half_axes_m", "endpoint_shape_exponent",
    "existence_logits", "geometry_uncertainty", "endpoint_evidence_logits",
])
def test_cached_geometry_rejects_nonfinite_prediction_fields(field):
    prediction, teacher = _cached_geometry()
    prediction[field].flat[0] = np.nan
    with pytest.raises(ValueError):
        score_cached_geometry(prediction, teacher)
