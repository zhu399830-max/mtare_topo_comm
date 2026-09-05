import numpy as np
import pytest

from mtare_topo.data.primitive_composition_anchor_sidecar import (
    canonical_anchor_digest,
    display_anchor_norm_histogram,
    materialize_current_sensor_anchors,
    validate_materialized_current_sensor_anchors,
)


def _batch():
    index = np.full((2, 32), -1, dtype=np.int32)
    mask = np.zeros((2, 32), dtype=np.uint8)
    index[:, :2] = (0, 1)
    mask[:, :2] = 1
    world = np.asarray(
        [
            [[2.0, 1.0, 3.0], [4.0, 1.0, 3.0]],
            [[2.0, 1.0, 3.0], [2.0, 5.0, 3.0]],
        ],
        dtype=np.float64,
    )
    sensor = np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    yaw = np.asarray([0.0, 90.0])
    return index, mask, world, sensor, yaw


def test_materialization_is_current_sensor_frame_and_zero_inactive():
    index, mask, world, sensor, yaw = _batch()
    value = materialize_current_sensor_anchors(
        primitive_index=index,
        primitive_mask=mask,
        anchor_world_m=world,
        sensor_xyz_m=sensor,
        yaw_deg=yaw,
    )
    assert value.dtype == np.float32
    np.testing.assert_array_equal(value[0, 0, 0], [1.0, 0.0, 2.0])
    np.testing.assert_allclose(value[1, 0, 0], [0.0, -1.0, 2.0], atol=1e-7)
    assert np.count_nonzero(value[:, 2:]) == 0


def test_same_construction_anchor_remains_exactly_equal():
    index, mask, world, sensor, yaw = _batch()
    value = materialize_current_sensor_anchors(
        primitive_index=index,
        primitive_mask=mask,
        anchor_world_m=world,
        sensor_xyz_m=sensor,
        yaw_deg=yaw,
    )
    np.testing.assert_array_equal(value[:, 0, 0], value[:, 1, 0])


def test_materialization_is_deterministic_and_digest_binds_sequence():
    index, mask, world, sensor, yaw = _batch()
    first = materialize_current_sensor_anchors(
        primitive_index=index, primitive_mask=mask, anchor_world_m=world,
        sensor_xyz_m=sensor, yaw_deg=yaw,
    )
    second = materialize_current_sensor_anchors(
        primitive_index=index, primitive_mask=mask, anchor_world_m=world,
        sensor_xyz_m=sensor, yaw_deg=yaw,
    )
    np.testing.assert_array_equal(first, second)
    source = np.asarray([5, 9], dtype=np.int64)
    assert canonical_anchor_digest(first, source) == canonical_anchor_digest(second, source)
    assert canonical_anchor_digest(first, source) != canonical_anchor_digest(second, source[::-1])


def test_invalid_index_contract_fails_closed():
    index, mask, world, sensor, yaw = _batch()
    index[0, 2] = 0
    with pytest.raises(ValueError, match="inactive"):
        materialize_current_sensor_anchors(
            primitive_index=index, primitive_mask=mask, anchor_world_m=world,
            sensor_xyz_m=sensor, yaw_deg=yaw,
        )


def test_nonzero_inactive_storage_fails_closed():
    index, mask, world, sensor, yaw = _batch()
    value = materialize_current_sensor_anchors(
        primitive_index=index, primitive_mask=mask, anchor_world_m=world,
        sensor_xyz_m=sensor, yaw_deg=yaw,
    )
    value[0, 3, 0, 0] = 1.0
    with pytest.raises(ValueError, match="inactive"):
        validate_materialized_current_sensor_anchors(value, mask)


def test_out_of_range_active_index_fails_closed():
    index, mask, world, sensor, yaw = _batch()
    index[0, 0] = len(world)
    with pytest.raises(ValueError, match="out of range"):
        materialize_current_sensor_anchors(
            primitive_index=index, primitive_mask=mask, anchor_world_m=world,
            sensor_xyz_m=sensor, yaw_deg=yaw,
        )


def test_display_histogram_records_overflow_without_rejecting_target():
    values = np.asarray([1.0, 79.5, 80.0, 143.0], dtype=np.float64)
    counts, overflow, maximum = display_anchor_norm_histogram(
        values, np.linspace(0.0, 80.0, 161),
    )
    assert int(np.sum(counts)) == len(values)
    assert overflow == 2
    assert maximum == 143.0
