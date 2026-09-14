"""Deterministic projection of seed-wise route-local structure centers."""

from __future__ import annotations

import numpy as np


def project_local_vectors(
    local_vector_m: np.ndarray,
    sensor_xyz_m: np.ndarray,
    route_local_basis: np.ndarray,
) -> np.ndarray:
    """Project route-local forward/left/up vectors into world coordinates."""

    vector = np.asarray(local_vector_m, dtype=np.float64)
    sensor = np.asarray(sensor_xyz_m, dtype=np.float64)
    basis = np.asarray(route_local_basis, dtype=np.float64)
    if vector.ndim != 2 or vector.shape[1] != 3:
        raise ValueError("local structure-center vectors must have shape [N,3]")
    if sensor.shape != vector.shape or basis.shape != (len(vector), 3, 3):
        raise ValueError("structure-center projection population drift")
    if not all(np.all(np.isfinite(value)) for value in (vector, sensor, basis)):
        raise ValueError("structure-center projection contains non-finite values")
    center = sensor + np.einsum("nij,ni->nj", basis, vector)
    if not np.all(np.isfinite(center)):
        raise ValueError("projected structure centers are non-finite")
    return center.astype(np.float32)


def combine_seed_projections(
    seed_local_vector_m: np.ndarray,
    sensor_xyz_m: np.ndarray,
    route_local_basis: np.ndarray,
) -> dict[str, np.ndarray]:
    """Combine three frozen seeds and expose an uncertainty radius.

    The uncertainty is the root-mean-square Euclidean distance from each seed
    center to the arithmetic ensemble center.  It is used only by the existing
    fail-closed deferred-association rule; it never enlarges the 4 m candidate
    radius.
    """

    vectors = np.asarray(seed_local_vector_m, dtype=np.float64)
    if vectors.ndim != 3 or vectors.shape[0] != 3 or vectors.shape[2] != 3:
        raise ValueError("seed local vectors must have shape [3,N,3]")
    centers = np.stack([
        project_local_vectors(value, sensor_xyz_m, route_local_basis)
        for value in vectors
    ]).astype(np.float64)
    ensemble_vector = vectors.mean(axis=0)
    ensemble_center = centers.mean(axis=0)
    direct_center = project_local_vectors(
        ensemble_vector, sensor_xyz_m, route_local_basis,
    ).astype(np.float64)
    linear_identity_max_abs_m = float(np.max(np.abs(ensemble_center - direct_center)))
    # Each seed center is intentionally rounded to the sealed float32 archive
    # before the arithmetic ensemble, matching qualification.  At world
    # coordinates up to roughly 600 m this introduces at most a few float32
    # ulps versus projecting the mean vector.  Keep a sub-millimetre bound.
    if linear_identity_max_abs_m > 1e-4:
        raise RuntimeError("linear structure-center ensemble identity drift")
    deviation = centers - ensemble_center[None]
    uncertainty = np.sqrt(np.mean(np.sum(deviation * deviation, axis=2), axis=0))
    if np.any(uncertainty > 12.0 + 1e-6) or not np.all(np.isfinite(uncertainty)):
        raise RuntimeError("structure-center seed uncertainty exceeds support")
    return {
        "seed_center_xyz_m": centers.astype(np.float32),
        "predicted_local_vector_m": ensemble_vector.astype(np.float32),
        "projected_center_xyz_m": ensemble_center.astype(np.float32),
        "position_uncertainty_m": uncertainty.astype(np.float32),
        "linear_identity_max_abs_m": np.asarray(linear_identity_max_abs_m, dtype=np.float64),
    }


__all__ = ["combine_seed_projections", "project_local_vectors"]
