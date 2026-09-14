"""Continuous spatial evidence, NOT a membership classifier or safety gate.

Consumes finite polylines and observed positions in one robot frame. No GT,
endpoint-as-opening assumption, learned score calibration, or new thresholds.
All per-segment alternatives survive; a nearest segment is not an identity.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CandidateSegmentEvidence:
    candidate_index: int
    primitive_index: int
    segment_index: int
    closest_xyz_m: tuple
    offset_xyz_m: tuple
    distance_m: float
    unoriented_angle_deg: float
    projection_clamped: bool


@dataclass(frozen=True)
class PrimitiveSpatialEvidence:
    candidate_segments: tuple
    # Pair entries: primitive indices, centroid displacement in current frame.
    relative_centers: tuple
    membership: None = None
    connectivity_verified: bool = False


def spatial_evidence(axes, primitive_indices, candidate_positions, candidate_directions):
    """No search radius, hard assignment, discarded candidates or inferred edge."""
    a = np.asarray(axes, dtype=float)
    p = np.asarray(candidate_positions, dtype=float)
    d = np.asarray(candidate_directions, dtype=float)
    ids = tuple(primitive_indices)
    if (a.ndim != 3 or a.shape[1:] != (3, 3) or p.ndim != 2 or p.shape[1:] != (3,)
            or d.shape != p.shape or len(ids) != len(a) or len(set(ids)) != len(ids)
            or any(type(i) is not int or i < 0 for i in ids)
            or not all(np.isfinite(v).all() for v in (a, p, d))):
        raise ValueError('finite same-frame three-control axes, unique indices and candidate vectors required')
    lengths = np.linalg.norm(np.diff(a, axis=1), axis=-1)
    if np.any(lengths <= 0) or not np.allclose(np.linalg.norm(d, axis=1), 1., atol=1e-8, rtol=0):
        raise ValueError('nondegenerate finite segments and unit directions required')
    rows = []
    for c, (point, direction) in enumerate(zip(p, d)):
        for identity, axis in zip(ids, a):
            for s, (start, end) in enumerate(zip(axis, axis[1:])):
                v = end-start
                t = float((point-start)@v/(v@v))
                nearest = start+np.clip(t, 0., 1.)*v
                offset = nearest-point
                angle = float(np.degrees(np.arccos(np.clip(abs(v@direction)/np.linalg.norm(v), 0., 1.))))
                rows.append(CandidateSegmentEvidence(c, identity, s, tuple(nearest), tuple(offset),
                            float(np.linalg.norm(offset)), angle, bool(t < 0 or t > 1)))
    centers = a.mean(axis=1)
    pairs = tuple((ids[i], ids[j], tuple(centers[j]-centers[i]))
                  for i in range(len(a)) for j in range(len(a)) if i != j)
    return PrimitiveSpatialEvidence(tuple(rows), pairs)
