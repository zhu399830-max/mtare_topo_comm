"""Teacher-only analytic polyline/sphere intersections for local candidates.

The sphere is the approved 10m observation ROI, not a persistent structure.
Returned crossings are neither observed openings nor nodes. A downstream
producer must bind sampled source geometry and test surface/ray support.
No construction identity, crop endpoint, XY projection, or support-interval
fill is used. Tangencies and roots at source vertices remain explicit
ambiguities rather than being silently displaced or counted twice.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SphereCrossings:
    segment_indices: np.ndarray
    segment_fractions: np.ndarray
    positions_m: np.ndarray
    outward_tangents: np.ndarray
    source_arc_m: np.ndarray
    ambiguous_segment_indices: tuple[int, ...]
    radius_m: float = 10.
    observed_openings: bool = False
    persistent_nodes: bool = False


def roi_crossings(points_m, *, center_m):
    """Intersect every segment in a source polyline with the fixed ROI.

    Strict-interior segment roots only; boundary vertices/tangencies are
    reported ambiguous, and callers must not treat this output as complete
    while any ambiguity exists. Both entry and exit orient away from ROI.
    Reversing a polyline reverses traversal order but not outward directions.
    Polyline discretization error is the source's, not analytic-spline truth.
    """
    points = np.asarray(points_m, dtype=np.float64)
    center = np.asarray(center_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2 or center.shape != (3,):
        raise ValueError("3D source polyline and explicit ROI center required")
    if not np.isfinite(points).all() or not np.isfinite(center).all():
        raise ValueError("nonfinite source geometry")
    delta = np.diff(points, axis=0)
    length = np.linalg.norm(delta, axis=1)
    if not np.isfinite(length).all() or np.any(length == 0):
        raise ValueError("zero or overflowing source segment; do not silently remove")
    arc = np.concatenate(([0.], np.cumsum(length)))
    indices, fractions, positions, normals, arcs, ambiguous = [], [], [], [], [], []
    for index, (point, step, size) in enumerate(zip(points[:-1], delta, length)):
        relative = point - center
        # Centering at the closest point on the infinite line avoids the
        # catastrophic discriminant subtraction for long near-tangent lines.
        direction = step / size
        along = -float(np.dot(relative, direction))
        nearest = relative + along * direction
        residual = 100. - float(np.dot(nearest, nearest))
        numerical = 128 * np.finfo(np.float64).eps * max(100., float(np.dot(relative, relative)), size*size)
        if not np.isfinite(residual) or not np.isfinite(numerical):
            raise ValueError("source geometry exceeds finite arithmetic")
        if residual < -numerical:
            continue
        if abs(residual) <= numerical:
            if -np.sqrt(numerical) <= along <= size + np.sqrt(numerical):
                ambiguous.append(index)
            continue
        offset = np.sqrt(residual)
        for distance in (along-offset, along+offset):
            fraction = distance/size
            eps = 128*np.finfo(np.float64).eps * max(1., abs(along/size), abs(offset/size))
            if abs(fraction) <= eps or abs(fraction-1.) <= eps:
                ambiguous.append(index)
                continue
            if not 0 < fraction < 1:
                continue
            position = point + fraction*step
            oriented = direction if np.dot(position-center, direction) > 0 else -direction
            indices.append(index); fractions.append(fraction); positions.append(position)
            normals.append(oriented); arcs.append(arc[index]+distance)
    def readonly(value, shape, dtype):
        result = np.asarray(value, dtype=dtype).reshape(shape)
        result.setflags(write=False)
        return result
    n = len(indices)
    return SphereCrossings(readonly(indices,(n,),np.int64),readonly(fractions,(n,),np.float64),
        readonly(positions,(n,3),np.float64),readonly(normals,(n,3),np.float64),
        readonly(arcs,(n,),np.float64),tuple(sorted(set(ambiguous))))
