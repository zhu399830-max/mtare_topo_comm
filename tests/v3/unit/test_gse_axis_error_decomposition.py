"""Three-control, two-segment diagnostics, not full splines or deployment scores."""
import numpy as np
import pytest

from mtare_topo.evaluation.gse_axis_error_decomposition import decompose_axes


def _line(half_length=2.0):
    return np.array([[-half_length, 0., 0.], [0., 0., 0.], [half_length, 0., 0.]])


def _identity(result):
    assert result["total_rms_m"] ** 2 == pytest.approx(
        result["axial_rms_m"] ** 2 + result["transverse_rms_m"] ** 2, abs=1e-10)


def test_identical_three_control_polylines_have_zero_error():
    result = decompose_axes(_line(), _line())
    for key in ("point_mean_m", "coordinate_mae_m", "axial_rms_m", "transverse_rms_m",
                "total_rms_m", "direction_error_deg", "undirected_direction_error_deg",
                "pred_to_teacher_polyline_m", "teacher_to_pred_polyline_m", "polyline_symmetric_m"):
        assert result[key] == pytest.approx(0., abs=1e-12)
    assert result["teacher_length_m"] == result["predicted_length_m"] == 4
    assert result["length_ratio"] == 1
    assert result["teacher_direction_resolved"] is True
    assert result["prediction_direction_resolved"] is True
    assert result["quadrature_error_bound_m"] == 8 / (4 * 256)
    _identity(result)


def test_parallel_three_metre_transverse_shift_is_not_axial_error():
    result = decompose_axes(_line() + (0, 3, 0), _line())
    assert result["point_mean_m"] == result["transverse_rms_m"] == result["total_rms_m"] == 3
    assert result["axial_rms_m"] == 0
    assert result["coordinate_mae_m"] == 1
    assert result["direction_error_deg"] == 0
    for key in ("pred_to_teacher_polyline_m", "teacher_to_pred_polyline_m", "polyline_symmetric_m"):
        assert result[key] == pytest.approx(3.)
    _identity(result)


def test_straight_crop_disagreement_is_axial_not_a_three_dimensional_offset():
    result = decompose_axes(_line(2), _line(4))
    assert result["point_mean_m"] == pytest.approx(4 / 3)
    assert result["axial_rms_m"] == pytest.approx(np.sqrt(8 / 3))
    assert result["transverse_rms_m"] == 0
    assert result["length_ratio"] == .5
    # The prediction lies on the teacher, but misses finite ends: not an infinite-line distance.
    assert result["pred_to_teacher_polyline_m"] == pytest.approx(0, abs=1e-12)
    assert result["teacher_to_pred_polyline_m"] == pytest.approx(.5)
    assert result["polyline_symmetric_m"] == pytest.approx(.25)
    _identity(result)


def test_ninety_degree_rotation_has_distinct_oriented_layout():
    predicted = _line()[:, [1, 0, 2]]
    result = decompose_axes(predicted, _line())
    assert result["direction_error_deg"] == pytest.approx(90.)
    assert result["undirected_direction_error_deg"] == pytest.approx(90.)
    assert result["point_mean_m"] == pytest.approx(4 * np.sqrt(2) / 3)
    assert result["axial_rms_m"] == pytest.approx(np.sqrt(8 / 3))
    assert result["transverse_rms_m"] == pytest.approx(np.sqrt(8 / 3))
    assert result["polyline_symmetric_m"] == pytest.approx(1.)
    _identity(result)


def test_reversal_is_not_silently_rematched_inside_diagnostic():
    result = decompose_axes(_line()[::-1], _line())
    assert result["point_mean_m"] == pytest.approx(8 / 3)
    assert result["direction_error_deg"] == pytest.approx(180.)
    assert result["undirected_direction_error_deg"] == pytest.approx(0.)
    assert result["polyline_symmetric_m"] == pytest.approx(0., abs=1e-12)
    assert result["transverse_rms_m"] == 0
    _identity(result)


def test_middle_control_is_used_as_two_finite_segments_not_a_chord_or_full_spline():
    teacher = np.array([[0., 0., 0.], [1., 1., 0.], [2., 0., 0.]])
    prediction = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    result = decompose_axes(prediction, teacher)
    assert result["teacher_length_m"] == pytest.approx(2 * np.sqrt(2))
    assert result["predicted_length_m"] == 2
    assert result["direction_error_deg"] == 0
    assert result["pred_to_teacher_polyline_m"] == pytest.approx(.5 / np.sqrt(2))
    assert result["teacher_to_pred_polyline_m"] == pytest.approx(.5)
    assert result["polyline_symmetric_m"] > 0


def test_common_rigid_transform_preserves_euclidean_diagnostics_but_not_coordinate_l1():
    teacher = np.array([[-3., 0., -1.], [0., 1., .3], [3., 2., 1.]])
    prediction = teacher + np.array([[.1, 1., -.3], [-.7, -.2, .5], [1.2, .4, .2]])
    v = np.array([1., 2., 3.]); v /= np.linalg.norm(v)
    cross = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    angle = .73
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    offset = np.array([10., -20., 7.])
    original = decompose_axes(prediction, teacher)
    transformed = decompose_axes(prediction @ rotation.T + offset, teacher @ rotation.T + offset)
    for key, value in original.items():
        if key not in ("coordinate_mae_m", "sensor_vertical_rms_m"):
            assert transformed[key] == pytest.approx(value, abs=1e-8), key
    residual = (prediction - teacher) @ rotation.T
    assert transformed["coordinate_mae_m"] == pytest.approx(np.abs(residual).mean())
    assert transformed["sensor_vertical_rms_m"] == pytest.approx(np.sqrt(np.square(residual[:, 2]).mean()))
    assert not np.isclose(transformed["coordinate_mae_m"], original["coordinate_mae_m"])
    _identity(transformed)


def test_zero_length_curves_preserve_sample_and_report_unresolved_direction():
    teacher = np.zeros((3, 3))
    predicted = np.tile((0., 3., 4.), (3, 1))
    result = decompose_axes(predicted, teacher)
    assert result["teacher_direction_resolved"] is False
    assert result["prediction_direction_resolved"] is False
    assert result["direction_error_deg"] is None
    assert result["undirected_direction_error_deg"] is None
    assert result["axial_rms_m"] is None and result["transverse_rms_m"] is None
    assert result["length_ratio"] is None
    assert result["teacher_length_m"] == result["predicted_length_m"] == 0
    assert result["point_mean_m"] == result["total_rms_m"] == 5
    assert result["polyline_symmetric_m"] == 5


def test_nonzero_polyline_with_zero_end_chord_does_not_invent_an_axis():
    teacher = np.array([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]])
    result = decompose_axes(_line(1), teacher)
    assert result["teacher_length_m"] == 2
    assert result["length_ratio"] == 1
    assert result["teacher_direction_resolved"] is False
    assert result["prediction_direction_resolved"] is True
    assert result["direction_error_deg"] is None
    assert result["axial_rms_m"] is None and result["transverse_rms_m"] is None


def test_quantized_away_displacement_does_not_produce_false_direction():
    # These perturbations are below float32 representability at this coordinate.
    points = np.array([[1e6, 0., 0.], [1e6 + .001, 0., 0.], [1e6 + .002, 0., 0.]], dtype=np.float32)
    result = decompose_axes(points, points)
    assert result["teacher_direction_resolved"] is False
    assert result["prediction_direction_resolved"] is False
    assert result["direction_error_deg"] is None
    assert result["point_mean_m"] == 0


def test_nonzero_but_one_source_ulp_chord_is_not_reliably_resolved():
    origin = np.float32(1e6)
    points = np.array([[origin, 0., 0.], [origin, 0., 0.],
                       [np.nextafter(origin, np.float32(np.inf)), 0., 0.]], dtype=np.float32)
    assert np.linalg.norm(points[-1] - points[0]) > 0
    result = decompose_axes(points, points)
    assert result["teacher_direction_resolved"] is False
    assert result["prediction_direction_resolved"] is False
    assert result["direction_error_deg"] is None
    assert result["teacher_length_m"] > 0
    assert result["length_ratio"] == 1


@pytest.mark.parametrize("duplicate_at", [0, 1])
def test_one_degenerate_segment_still_uses_remaining_finite_segment(duplicate_at):
    teacher = _line()
    teacher[duplicate_at + 1] = teacher[duplicate_at]
    result = decompose_axes(teacher, teacher)
    assert result["point_mean_m"] == 0
    assert result["polyline_symmetric_m"] == pytest.approx(0, abs=1e-12)
    assert result["teacher_direction_resolved"] is True


@pytest.mark.parametrize("samples", [0, -1, 1.5, True])
def test_invalid_quadrature_counts_are_rejected(samples):
    with pytest.raises(ValueError):
        decompose_axes(_line(), _line(), samples=samples)


@pytest.mark.parametrize("samples", [32, 64, 256])
def test_midpoint_distance_error_is_bounded_without_choosing_a_passing_resolution(samples):
    teacher = np.array([[-3., .2, 0.], [-.1, 1.7, .3], [2.8, -.4, 0.]])
    predicted = np.array([[-2.1, -.1, .8], [.7, .4, -.2], [2.2, 1.3, .4]])
    coarse = decompose_axes(predicted, teacher, samples=samples)
    fine = decompose_axes(predicted, teacher, samples=4096)
    assert coarse["quadrature_error_bound_m"] == pytest.approx(
        (coarse["predicted_length_m"] + coarse["teacher_length_m"]) / (4 * samples))
    assert abs(coarse["polyline_symmetric_m"] - fine["polyline_symmetric_m"]) <= (
        coarse["quadrature_error_bound_m"] + fine["quadrature_error_bound_m"])
