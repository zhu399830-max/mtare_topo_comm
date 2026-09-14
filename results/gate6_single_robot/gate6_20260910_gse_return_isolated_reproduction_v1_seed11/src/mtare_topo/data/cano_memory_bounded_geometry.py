"""Memory-bounded, result-equivalent helpers for the pinned Cano generator."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import numpy as np


DEFAULT_SCRATCH_LIMIT_BYTES = 32 * 1024**2
# The pinned implementation materializes several float64 A x B x 3 temporaries.
# This conservative accounting affects only chunk size, never geometry semantics.
ESTIMATED_TEMPORARY_BYTES_PER_PAIR = 96


def _chunk_rows(reference_count: int, scratch_limit_bytes: int) -> int:
    if reference_count <= 0:
        raise ValueError("reference_count must be positive")
    if scratch_limit_bytes <= 0:
        raise ValueError("scratch_limit_bytes must be positive")
    return max(
        1,
        scratch_limit_bytes
        // (reference_count * ESTIMATED_TEMPORARY_BYTES_PER_PAIR),
    )


def _nearest_reference(
    queries: np.ndarray,
    references: np.ndarray,
    *,
    scratch_limit_bytes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the pinned distance-matrix argmin/min without retaining all rows."""

    from subt_proc_gen.geometry import distance_matrix

    rows = _chunk_rows(len(references), scratch_limit_bytes)
    nearest = np.empty(len(queries), dtype=np.intp)
    distances = np.empty(len(queries), dtype=np.float64)
    for start in range(0, len(queries), rows):
        stop = min(start + rows, len(queries))
        matrix = distance_matrix(queries[start:stop], references)
        nearest[start:stop] = np.argmin(matrix, axis=1)
        distances[start:stop] = np.min(matrix, axis=1)
    return nearest, distances


def _selected_pair_distances(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Evaluate selected distance-matrix cells with the pinned arithmetic."""

    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 3:
        raise ValueError("selected point arrays must both have shape (N, 3)")
    return np.linalg.norm(np.ones(first.shape) * first - second, axis=1)


def points_inside_of_tunnel_section_bounded(
    axis_points: np.ndarray,
    tunnel_points: np.ndarray,
    points: np.ndarray,
    axis_vectors: np.ndarray | None = None,
    max_projection_over_axis: float = 0.5,
    *,
    scratch_limit_bytes: int = DEFAULT_SCRATCH_LIMIT_BYTES,
) -> tuple[np.ndarray]:
    """Pinned Cano containment test with bounded distance-matrix temporaries.

    Query rows are chunked, but every row still sees every reference in the
    original order.  Consequently ``argmin`` tie-breaking and distance values
    retain the original semantics.
    """

    try:
        assert axis_points.shape[0] > 0
        assert axis_points.shape[1] == 3
        assert tunnel_points.shape[0] > 0
        assert tunnel_points.shape[1] == 3
        assert points.shape[0] > 0
        assert points.shape[1] == 3
    except (AssertionError, IndexError):
        return (np.array([], dtype=int),)

    closest_ap_to_p_idx, dist_to_ap_of_p = _nearest_reference(
        points,
        axis_points,
        scratch_limit_bytes=scratch_limit_bytes,
    )
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

    closest_tp_to_p, _ = _nearest_reference(
        points,
        tunnel_points,
        scratch_limit_bytes=scratch_limit_bytes,
    )
    selected_tunnel_points = tunnel_points[closest_tp_to_p, :]
    selected_axis_points = axis_points[closest_ap_to_p_idx, :]
    selected_radius = _selected_pair_distances(
        selected_tunnel_points, selected_axis_points
    )
    inside_for_radius = dist_to_ap_of_p < selected_radius
    inside = np.logical_and(
        inside_for_radius,
        np.logical_not(outside_because_of_angle),
    )
    return np.where(inside)


def ids_points_inside_ptcl_sphere_bounded(
    sphere_points: np.ndarray,
    center_point: np.ndarray,
    points: np.ndarray,
    *,
    scratch_limit_bytes: int = DEFAULT_SCRATCH_LIMIT_BYTES,
) -> tuple[np.ndarray]:
    """Pinned Cano point-cloud sphere containment with bounded temporaries."""

    closest_sphere_point, _ = _nearest_reference(
        points,
        sphere_points,
        scratch_limit_bytes=scratch_limit_bytes,
    )
    sphere_radius = np.reshape(
        np.linalg.norm(sphere_points - center_point, axis=1), (-1, 1)
    )
    point_radius = np.reshape(
        np.linalg.norm(points - center_point, axis=1), (-1, 1)
    )
    return np.where(point_radius < sphere_radius[closest_sphere_point])


@contextmanager
def patched_cano_points_inside() -> Iterator[None]:
    """Temporarily install the bounded helper in the pinned generator module."""

    import subt_proc_gen.mesh_generation as mesh_generation

    original_tunnel = mesh_generation.points_inside_of_tunnel_section
    original_sphere = mesh_generation.ids_points_inside_ptcl_sphere
    mesh_generation.points_inside_of_tunnel_section = (
        points_inside_of_tunnel_section_bounded
    )
    mesh_generation.ids_points_inside_ptcl_sphere = (
        ids_points_inside_ptcl_sphere_bounded
    )
    try:
        yield
    finally:
        mesh_generation.points_inside_of_tunnel_section = original_tunnel
        mesh_generation.ids_points_inside_ptcl_sphere = original_sphere
