"""Objective signed structure-center targets for route-conditioned hypotheses."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


DECISION_EVENTS = frozenset(("junction", "terminal"))


def route_local_basis(route_tangent_xyz: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Build deterministic forward/lateral/up bases for valid route rows."""

    tangent = np.asarray(route_tangent_xyz, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if tangent.ndim != 2 or tangent.shape[1] != 3 or valid.shape != (len(tangent),):
        raise ValueError("route-local basis population drift")
    basis = np.zeros((len(tangent), 3, 3), dtype=np.float64)
    for row in np.flatnonzero(valid):
        norm = float(np.linalg.norm(tangent[row]))
        if not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError("valid route-local tangent is degenerate")
        forward = tangent[row] / norm
        lateral = np.asarray([-forward[1], forward[0], 0.0])
        lateral_norm = float(np.linalg.norm(lateral))
        if lateral_norm <= 1e-8:
            raise ValueError("route-local lateral axis is degenerate")
        lateral /= lateral_norm
        up = np.cross(forward, lateral)
        basis[row] = np.stack((forward, lateral, up))
    return basis.astype(np.float32)


def local_event_center_vectors(
    sensor_xyz_m: np.ndarray, objective_center_xyz_m: np.ndarray,
    route_tangent_xyz: np.ndarray, valid_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Express objective event centers in an online-constructible local frame."""

    sensor = np.asarray(sensor_xyz_m, dtype=np.float64)
    center = np.asarray(objective_center_xyz_m, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if sensor.shape != center.shape or sensor.shape != (len(valid), 3):
        raise ValueError("local event-center source drift")
    basis = route_local_basis(route_tangent_xyz, valid).astype(np.float64)
    vector = np.zeros_like(sensor)
    delta = center[valid] - sensor[valid]
    vector[valid] = np.einsum("nij,nj->ni", basis[valid], delta)
    if np.any(np.abs(vector[valid, 0]) > 12.0 + 1e-6) or np.any(np.abs(vector[valid, 1:]) > 5.0 + 1e-6):
        raise ValueError("local event-center vector exceeds frozen support")
    reconstructed = np.einsum("nij,ni->nj", basis[valid], vector[valid]) + sensor[valid]
    if not np.allclose(reconstructed, center[valid], atol=1e-5):
        raise ValueError("local event-center reconstruction drift")
    return {"route_local_basis": basis.astype(np.float32), "local_center_vector_m": vector.astype(np.float32)}


def traversal_tangents(
    traversal_id: Sequence[str], sequence_index: Sequence[int], xyz_m: np.ndarray,
) -> np.ndarray:
    """Estimate a causal-route tangent without crossing traversal boundaries."""

    traversal = np.asarray(traversal_id, dtype=str)
    sequence = np.asarray(sequence_index, dtype=np.int64)
    xyz = np.asarray(xyz_m, dtype=np.float64)
    if traversal.ndim != 1 or sequence.shape != traversal.shape or xyz.shape != (len(traversal), 3):
        raise ValueError("event-center tangent population drift")
    lookup: dict[tuple[str, int], int] = {}
    for row, (name, value) in enumerate(zip(traversal, sequence, strict=True)):
        key = (str(name), int(value))
        if value < 0 or key in lookup:
            raise ValueError("event-center traversal identity is invalid")
        lookup[key] = row
    tangent = np.empty_like(xyz)
    for row, (name, value) in enumerate(zip(traversal, sequence, strict=True)):
        previous = lookup.get((str(name), int(value) - 1))
        following = lookup.get((str(name), int(value) + 1))
        if previous is not None and following is not None:
            delta = xyz[following] - xyz[previous]
        elif following is not None:
            delta = xyz[following] - xyz[row]
        elif previous is not None:
            delta = xyz[row] - xyz[previous]
        else:
            # A sealed C07 world has two one-observation corridor traversals.
            # They are never offset-supervision rows, so retain an explicit
            # zero sentinel instead of inventing a direction from Teacher
            # geometry. A decision row with this sentinel still fails below.
            tangent[row] = 0.0
            continue
        norm = float(np.linalg.norm(delta))
        if not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError("event-center traversal tangent is degenerate")
        tangent[row] = delta / norm
    return tangent.astype(np.float32)


def event_center_targets(
    rows: Sequence[Mapping[str, object]],
    sensor_xyz_m: np.ndarray,
    node_xyz_by_world: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, np.ndarray]:
    """Create signed longitudinal targets; objective identity is teacher-only."""

    xyz = np.asarray(sensor_xyz_m, dtype=np.float64)
    if len(rows) == 0 or xyz.shape != (len(rows), 3) or not np.all(np.isfinite(xyz)):
        raise ValueError("event-center source population drift")
    traversal = np.asarray([str(row["traversal_id"]) for row in rows])
    sequence = np.asarray([int(row["sequence_index"]) for row in rows], dtype=np.int64)
    tangent = traversal_tangents(traversal, sequence, xyz).astype(np.float64)
    valid = np.zeros(len(rows), dtype=np.bool_)
    offset = np.zeros(len(rows), dtype=np.float32)
    center = np.zeros((len(rows), 3), dtype=np.float32)
    projected = np.zeros((len(rows), 3), dtype=np.float32)
    transverse = np.zeros(len(rows), dtype=np.float32)
    for index, row in enumerate(rows):
        event = str(row["event"])
        identity = row.get("identity")
        if event not in DECISION_EVENTS:
            # Geometry-transition labels may be anchored to an objective node,
            # but they are edge-geometry events in the factorized graph and
            # therefore remain explicitly masked from center-offset loss.
            continue
        if float(np.linalg.norm(tangent[index])) < 0.5:
            raise ValueError("decision event has no deployment-valid traversal tangent")
        prefix = f"{row['parent_id']}:node:"
        if identity is None or not str(identity).startswith(prefix):
            raise ValueError("decision event lacks a typed node identity")
        node_id = str(identity)[len(prefix):]
        try:
            objective = np.asarray(node_xyz_by_world[str(row["parent_id"])][node_id], dtype=np.float64)
        except KeyError as exc:
            raise ValueError("decision identity is absent from its objective graph") from exc
        if objective.shape != (3,) or not np.all(np.isfinite(objective)):
            raise ValueError("objective event center is invalid")
        delta = objective - xyz[index]
        signed = float(np.dot(delta, tangent[index]))
        projection = xyz[index] + signed * tangent[index]
        residual = float(np.linalg.norm(objective - projection))
        if abs(signed) > 12.0 + 1e-6 or residual > 5.0 + 1e-6:
            raise ValueError("event-center target exceeds the frozen geometric support")
        valid[index] = True
        offset[index] = signed
        center[index] = objective
        projected[index] = projection
        transverse[index] = residual
    if not np.any(valid):
        raise ValueError("event-center target population is empty")
    return {
        "valid_mask": valid,
        "route_tangent_xyz": tangent.astype(np.float32),
        "signed_center_offset_m": offset,
        "objective_center_xyz_m": center,
        "oracle_longitudinal_center_xyz_m": projected,
        "transverse_residual_m": transverse,
    }


__all__ = ["DECISION_EVENTS", "event_center_targets", "traversal_tangents", "route_local_basis", "local_event_center_vectors"]
