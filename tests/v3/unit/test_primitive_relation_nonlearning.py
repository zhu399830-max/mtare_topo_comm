from __future__ import annotations

import math

import numpy as np
import pytest

from mtare_topo.semantics.primitive_relation_nonlearning import (
    RobustPrimitiveRelationBaseline,
)


def _window(headings: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranges = np.full((5, 16, 720), 5.0 / 50.0, dtype=np.float32)
    valid = np.ones_like(ranges)
    columns = np.arange(720, dtype=np.float64) * 0.5
    for heading in headings:
        distance = np.abs((columns - heading + 180.0) % 360.0 - 180.0)
        ranges[:, :, distance <= 6.0] = 25.0 / 50.0
    range_valid = np.stack((ranges, valid), axis=1)
    translation = np.zeros((5, 3), dtype=np.float32)
    translation[:, 0] = np.arange(-4, 1, dtype=np.float32)
    yaw = np.zeros(5, dtype=np.float32)
    return range_valid, translation, yaw


def test_opposite_corridor_sectors_form_one_undirected_primitive() -> None:
    prediction = RobustPrimitiveRelationBaseline().predict(*_window((0.0, 180.0)))
    assert int(prediction.primitive_mask.sum()) == 1
    assert np.all(prediction.endpoint_half_axes_m[0] > 0.0)
    assert np.all((prediction.endpoint_shape_exponent[0] >= 2.0) & (prediction.endpoint_shape_exponent[0] <= 10.0))
    assert np.all(prediction.temporal_visibility[:, 0] == 1)


def test_three_way_opening_keeps_stubs_and_symmetric_relations() -> None:
    prediction = RobustPrimitiveRelationBaseline().predict(*_window((0.0, 90.0, 180.0)))
    assert int(prediction.primitive_mask.sum()) == 3
    np.testing.assert_array_equal(
        prediction.endpoint_attachment,
        prediction.endpoint_attachment.transpose(2, 3, 0, 1),
    )
    np.testing.assert_array_equal(prediction.disconnected_overlap, prediction.disconnected_overlap.T)
    assert int(prediction.endpoint_attachment.sum()) > 0


def test_baseline_is_deterministic() -> None:
    method = RobustPrimitiveRelationBaseline(); inputs = _window((20.0, 140.0, 260.0))
    first = method.predict(*inputs); second = method.predict(*inputs)
    for name in first.__dict__:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_rotation_equivariance_for_corridor_axis() -> None:
    method = RobustPrimitiveRelationBaseline(); values, translation, yaw = _window((0.0, 180.0))
    first = method.predict(values, translation, yaw)
    shift = 72; angle = math.radians(36.0)
    rotated_values = np.roll(values, shift, axis=-1)
    rotation = np.asarray(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    rotated_translation = translation.copy(); rotated_translation[:, :2] = translation[:, :2] @ rotation.T
    second = method.predict(rotated_values, rotated_translation, yaw)
    expected = first.axis_control_current_sensor_m[0].copy()
    expected[:, :2] = expected[:, :2] @ rotation.T
    direct = np.max(np.abs(second.axis_control_current_sensor_m[0] - expected))
    reversed_error = np.max(np.abs(second.axis_control_current_sensor_m[0] - expected[::-1]))
    assert min(float(direct), float(reversed_error)) <= 0.15


def test_invalid_student_input_fails_closed() -> None:
    values, translation, yaw = _window((0.0, 180.0)); method = RobustPrimitiveRelationBaseline()
    with pytest.raises(ValueError, match="expects"):
        method.predict(values[..., :-1], translation, yaw)
    translation[-1, 0] = 1.0
    with pytest.raises(ValueError, match="current relative odometry"):
        method.predict(values, translation, yaw)
