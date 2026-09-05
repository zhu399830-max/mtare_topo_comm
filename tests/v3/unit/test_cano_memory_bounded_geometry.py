from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL = PROJECT_ROOT / "external/procedural-subt-gen/src"
if str(EXTERNAL) not in sys.path:
    sys.path.insert(0, str(EXTERNAL))

from mtare_topo.data.cano_memory_bounded_geometry import (  # noqa: E402
    _chunk_rows,
    _nearest_reference,
    _selected_pair_distances,
    ids_points_inside_ptcl_sphere_bounded,
    patched_cano_points_inside,
    points_inside_of_tunnel_section_bounded,
)
from subt_proc_gen.geometry import distance_matrix  # noqa: E402


def _pinned_points_inside(
    axis_points: np.ndarray,
    tunnel_points: np.ndarray,
    points: np.ndarray,
    axis_vectors: np.ndarray | None = None,
    max_projection_over_axis: float = 0.5,
) -> tuple[np.ndarray]:
    dist_of_ps_to_aps = distance_matrix(points, axis_points)
    closest_ap_to_p_idx = np.argmin(dist_of_ps_to_aps, axis=1)
    if axis_vectors is not None:
        closest_ap_to_p_vector = points - axis_points[closest_ap_to_p_idx, :]
        ap_vector_of_p = axis_vectors[closest_ap_to_p_idx, :]
        projection = np.array(
            [
                np.dot(ap_vector_of_p[i, :], closest_ap_to_p_vector[i, :])
                for i in range(ap_vector_of_p.shape[0])
            ]
        )
        outside_because_of_angle = (
            np.abs(projection) > max_projection_over_axis
        )
    else:
        outside_because_of_angle = np.zeros(points.shape[0], dtype=np.bool_)
    dist_of_ps_to_tps = distance_matrix(points, tunnel_points)
    dist_of_tps_to_aps = distance_matrix(tunnel_points, axis_points)
    dist_to_ap_of_p = np.min(dist_of_ps_to_aps, axis=1)
    closest_tp_to_p = np.argmin(dist_of_ps_to_tps, axis=1)
    inside_for_radius = (
        dist_to_ap_of_p
        < dist_of_tps_to_aps[closest_tp_to_p, closest_ap_to_p_idx]
    )
    return np.where(
        np.logical_and(inside_for_radius, np.logical_not(outside_because_of_angle))
    )


def _pinned_points_inside_sphere(
    sphere_points: np.ndarray,
    center_point: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray]:
    distances = distance_matrix(points, sphere_points)
    sphere_radius = np.reshape(
        np.linalg.norm(sphere_points - center_point, axis=1), (-1, 1)
    )
    point_radius = np.reshape(
        np.linalg.norm(points - center_point, axis=1), (-1, 1)
    )
    closest = np.argmin(distances, axis=1)
    return np.where(point_radius < sphere_radius[closest])


def test_nearest_reference_matches_full_matrix_exactly_across_chunks() -> None:
    rng = np.random.default_rng(20260822)
    queries = rng.normal(size=(37, 3))
    references = rng.normal(size=(19, 3))
    matrix = distance_matrix(queries, references)
    for scratch in (1, 19 * 96, 19 * 96 * 7, 10**9):
        nearest, distances = _nearest_reference(
            queries, references, scratch_limit_bytes=scratch
        )
        assert np.array_equal(nearest, np.argmin(matrix, axis=1))
        assert np.array_equal(distances, np.min(matrix, axis=1))


def test_tie_breaking_keeps_first_reference() -> None:
    queries = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    references = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    nearest, _ = _nearest_reference(
        queries, references, scratch_limit_bytes=1
    )
    assert nearest.tolist() == [0, 1]


def test_selected_pair_distances_equal_full_matrix_cells() -> None:
    rng = np.random.default_rng(97)
    for dtype in (np.float32, np.float64):
        first = rng.normal(size=(23, 3)).astype(dtype)
        second = rng.normal(size=(17, 3)).astype(dtype)
        first_ids = rng.integers(0, len(first), size=41)
        second_ids = rng.integers(0, len(second), size=41)
        expected = distance_matrix(first, second)[first_ids, second_ids]
        observed = _selected_pair_distances(
            first[first_ids], second[second_ids]
        )
        assert np.array_equal(observed, expected)


def test_points_inside_matches_pinned_function_with_and_without_vectors() -> None:
    rng = np.random.default_rng(53)
    axis_points = rng.normal(size=(13, 3))
    tunnel_points = rng.normal(size=(31, 3))
    points = rng.normal(size=(41, 3))
    axis_vectors = rng.normal(size=(13, 3))
    for vectors in (None, axis_vectors):
        expected = _pinned_points_inside(
            axis_points, tunnel_points, points, vectors
        )
        for scratch in (1, 31 * 96 * 3, 10**9):
            observed = points_inside_of_tunnel_section_bounded(
                axis_points,
                tunnel_points,
                points,
                vectors,
                scratch_limit_bytes=scratch,
            )
            assert len(observed) == len(expected) == 1
            assert np.array_equal(observed[0], expected[0])


def test_points_inside_preserves_duplicates_and_strict_radius_boundary() -> None:
    axis = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    tunnel = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    points = np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    expected = _pinned_points_inside(axis, tunnel, points)
    observed = points_inside_of_tunnel_section_bounded(
        axis, tunnel, points, scratch_limit_bytes=1
    )
    assert np.array_equal(observed[0], expected[0])
    assert observed[0].tolist() == [0]


def test_invalid_shapes_preserve_empty_tuple_contract() -> None:
    invalid = np.empty((0, 3))
    valid = np.ones((2, 3))
    observed = points_inside_of_tunnel_section_bounded(invalid, valid, valid)
    assert len(observed) == 1
    assert observed[0].dtype.kind in "iu"
    assert observed[0].size == 0


def test_chunk_size_is_positive_and_scratch_bounded() -> None:
    assert _chunk_rows(1_000_000, 32 * 1024**2) == 1
    assert _chunk_rows(10, 10 * 96 * 7) == 7


def test_sphere_containment_matches_pinned_function() -> None:
    rng = np.random.default_rng(71)
    sphere_points = rng.normal(size=(29, 3))
    center = rng.normal(size=3)
    points = rng.normal(size=(47, 3))
    expected = _pinned_points_inside_sphere(sphere_points, center, points)
    for scratch in (1, 29 * 96 * 5, 10**9):
        observed = ids_points_inside_ptcl_sphere_bounded(
            sphere_points, center, points, scratch_limit_bytes=scratch
        )
        assert np.array_equal(observed[0], expected[0])


def test_patch_is_scoped_and_restores_original(monkeypatch) -> None:
    original_tunnel = object()
    original_sphere = object()
    mesh_generation = SimpleNamespace(
        points_inside_of_tunnel_section=original_tunnel,
        ids_points_inside_ptcl_sphere=original_sphere,
    )
    monkeypatch.setitem(
        sys.modules, "subt_proc_gen.mesh_generation", mesh_generation
    )
    with patched_cano_points_inside():
        assert (
            mesh_generation.points_inside_of_tunnel_section
            is points_inside_of_tunnel_section_bounded
        )
        assert (
            mesh_generation.ids_points_inside_ptcl_sphere
            is ids_points_inside_ptcl_sphere_bounded
        )
    assert mesh_generation.points_inside_of_tunnel_section is original_tunnel
    assert mesh_generation.ids_points_inside_ptcl_sphere is original_sphere
