"""Synthetic convex-support certificates; no claim about learned assignment."""
from itertools import product

import numpy as np
import pytest

from mtare_topo.evaluation.gse_coordinate_support import coordinate_support_bounds


def _verify(points, queries, result):
    points, queries = np.asarray(points, dtype=np.float64), np.asarray(queries, dtype=np.float64)
    lower, upper = result["lower_bound_m"], result["upper_bound_m"]
    assert lower.shape == upper.shape == (len(queries),)
    assert np.isfinite(lower).all() and np.isfinite(upper).all()
    assert np.all(lower >= 0) and np.all(upper >= 0)
    assert np.all(lower <= upper + 1e-12)
    bound = result["numerical_bound_m"]
    assert np.isfinite(bound) and bound >= 0
    assert result["affine_rank"] in (0, 1, 2, 3)
    assert isinstance(result["hull_status"], str)
    assert len(result["status"]) == len(result["witnesses"]) == len(queries)
    if "lower_certificate_directions" in result:
        directions = np.asarray(result["lower_certificate_directions"])
        support_indices = np.asarray(result["lower_certificate_support_indices"])
        assert directions.shape == queries.shape
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-12)
        for row, direction in enumerate(directions):
            support = points[support_indices[row]]
            assert np.max((points - support) @ direction) <= bound + 1e-10
            certified = max(0., (queries[row] - support) @ direction - bound)
            assert lower[row] == pytest.approx(certified, abs=bound + 1e-10)
    for row, witness in enumerate(result["witnesses"]):
        assert result["status"][row] in {
            "OUTSIDE_CERTIFIED", "WITHIN_TOLERANCE_WITNESS", "UNRESOLVED"}
        center_weight = witness["center_weight"]
        indices = witness["point_indices"]
        weights = np.asarray(witness["point_weights"], dtype=np.float64)
        assert len(indices) == len(weights)
        assert all(isinstance(index, (int, np.integer)) and 0 <= index < len(points) for index in indices)
        assert np.isfinite(center_weight) and center_weight >= 0
        assert np.isfinite(weights).all() and np.all(weights >= 0)
        assert center_weight + weights.sum() == pytest.approx(1., abs=1e-12)
        reconstructed = center_weight * points.mean(axis=0)
        if indices:
            reconstructed += np.sum(points[indices] * weights[:, None], axis=0)
        residual = np.linalg.norm(reconstructed - queries[row])
        assert upper[row] + 1e-10 >= residual
        if result["status"][row] == "WITHIN_TOLERANCE_WITNESS":
            assert residual <= bound + 1e-10


def _run(points, queries):
    points, queries = np.asarray(points), np.asarray(queries)
    result = coordinate_support_bounds(points, queries)
    _verify(points, queries, result)
    return result


def test_single_point_rank_zero_exact_and_outside_queries():
    points = np.array([[1., 2., 3.]])
    result = _run(points, np.array([[1., 2., 3.], [1., 5., 7.]]))
    assert result["affine_rank"] == 0
    assert result["status"] == ["WITHIN_TOLERANCE_WITNESS", "OUTSIDE_CERTIFIED"]
    assert result["lower_bound_m"][1] <= 5 <= result["upper_bound_m"][1] + 1e-12
    assert result["upper_bound_m"][0] <= result["numerical_bound_m"] + 1e-12


def test_line_segment_requires_off_axis_and_finite_endpoint_checks():
    points = np.array([[-2., 0., 0.], [2., 0., 0.], [0., 0., 0.]])
    queries = np.array([[.5, 0., 0.], [3., 0., 0.], [.5, 3., 0.]])
    result = _run(points, queries)
    assert result["affine_rank"] == 1
    assert result["status"] == ["WITHIN_TOLERANCE_WITNESS", "OUTSIDE_CERTIFIED", "OUTSIDE_CERTIFIED"]
    distances = np.array([0., 1., 3.])
    assert np.all(result["lower_bound_m"] <= distances + 1e-10)
    assert np.all(result["upper_bound_m"] + 1e-10 >= distances)


def test_plane_square_corner_bound_is_not_claimed_as_exact_distance():
    points = np.array([[x, y, 0.] for x, y in product((-1., 1.), repeat=2)])
    queries = np.array([[.25, .5, 0.], [0., 0., 2.], [2., 2., 0.]])
    result = _run(points, queries)
    assert result["affine_rank"] == 2
    assert result["status"] == ["WITHIN_TOLERANCE_WITNESS", "OUTSIDE_CERTIFIED", "OUTSIDE_CERTIFIED"]
    assert result["lower_bound_m"][2] <= np.sqrt(2) + 1e-10
    assert result["upper_bound_m"][2] + 1e-10 >= np.sqrt(2)


def test_tetrahedron_interior_has_legal_witness_and_external_query_is_separated():
    points = np.array([[0., 0., 0.], [4., 0., 0.], [0., 4., 0.], [0., 0., 4.]])
    inside = np.array([.1, .2, .3, .4]) @ points
    result = _run(points, np.array([inside, [4., 4., 4.]]))
    assert result["affine_rank"] == 3
    assert result["status"] == ["WITHIN_TOLERANCE_WITNESS", "OUTSIDE_CERTIFIED"]
    assert result["lower_bound_m"][1] > 0


@pytest.mark.parametrize("thickness", [1e-6, 1e-12, 1e-16])
def test_thin_tetrahedron_rank_reduction_never_certifies_an_inside_point_outside(thickness):
    points = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [.5, .5, thickness]])
    queries = np.array([[.1, .2, .3, .4], [.01, .01, .01, .97]]) @ points
    result = _run(points, queries)
    assert "OUTSIDE_CERTIFIED" not in result["status"]
    assert np.all(result["lower_bound_m"] <= result["numerical_bound_m"] + 1e-10)


def test_many_nonuniform_known_convex_combinations_are_never_false_outside():
    rng = np.random.default_rng(17)
    points = rng.normal(size=(40, 3))
    weights = rng.uniform(size=(20, 40))
    weights /= weights.sum(axis=1, keepdims=True)
    result = _run(points, weights @ points)
    assert all(status == "WITHIN_TOLERANCE_WITNESS" for status in result["status"])


def test_coplanar_hull_triangles_do_not_hide_nonboundary_cube_interior_witnesses():
    points = np.array(list(product((-1., 1.), repeat=3)))
    queries = np.array(list(product((-.6, .2, .7), repeat=3)))
    result = _run(points, queries)
    assert all(status == "WITHIN_TOLERANCE_WITNESS" for status in result["status"])


def test_raw_cloud_contains_mean_token_hull_and_can_expose_pooling_loss():
    raw = np.array([[x, y, z] for y, z in product((-1., 1.), repeat=2) for x in (-1., 1.)])
    means = raw.reshape(4, 2, 3).mean(axis=1)
    queries = np.array([[.5, 0., 0.], [0., .3, -.2], [3., 0., 0.]])
    raw_result = _run(raw, queries)
    mean_result = _run(means, queries)
    assert raw_result["status"][0] == "WITHIN_TOLERANCE_WITNESS"
    assert mean_result["status"][0] == "OUTSIDE_CERTIFIED"
    # This interval inconsistency would contradict conv(means) subset conv(raw).
    error = raw_result["numerical_bound_m"] + mean_result["numerical_bound_m"]
    assert np.all(raw_result["lower_bound_m"] <= mean_result["upper_bound_m"] + error + 1e-10)


def test_duplicate_supports_and_permutation_keep_valid_nonboundary_certificates():
    points = np.array(list(product((-1., 1.), repeat=3)))
    queries = np.array([[.1, .2, .3], [3., .2, .1]])
    original = _run(points, queries)
    duplicated = np.concatenate((points, points[[0, 3, 3]]))
    shuffled = duplicated[np.array([10, 4, 1, 6, 0, 2, 3, 7, 5, 9, 8])]
    changed = _run(shuffled, queries)
    assert changed["status"] == original["status"]
    assert changed["affine_rank"] == original["affine_rank"]


def test_rigid_transform_retains_nonboundary_inside_and_outside_status():
    points = np.array(list(product((-1., 1.), repeat=3)))
    queries = np.array([[.1, .2, .3], [3., .2, .1]])
    angle = .47
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                         [-np.sin(angle), 0., np.cos(angle)]])
    original = _run(points, queries)
    transformed = _run(points @ rotation.T + (10, -7, 3), queries @ rotation.T + (10, -7, 3))
    assert transformed["status"] == original["status"]
    assert transformed["affine_rank"] == original["affine_rank"]
    assert transformed["upper_bound_m"] == pytest.approx(original["upper_bound_m"], abs=1e-9)


@pytest.mark.parametrize("case", ["nan_points", "inf_queries", "point_shape", "query_shape", "empty_points"])
def test_invalid_nonfinite_or_malformed_inputs_are_rejected(case):
    points = np.array([[0., 0., 0.], [1., 1., 1.]])
    queries = np.array([[.5, .5, .5]])
    if case == "nan_points":
        points[0, 0] = np.nan
    elif case == "inf_queries":
        queries[0, 1] = np.inf
    elif case == "point_shape":
        points = points[:, :2]
    elif case == "query_shape":
        queries = queries.reshape(3)
    else:
        points = points[:0]
    with pytest.raises(ValueError):
        coordinate_support_bounds(points, queries)
