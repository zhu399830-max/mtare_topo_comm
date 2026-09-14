"""Certified convex-support bounds, not learned fitting or topology evidence.

SVD/Qhull propose directions and convex witnesses only. Lower certificates
recompute support values on ALL original points; upper certificates reconstruct
a convex combination of original points. A discarded SVD direction is never
assumed to have zero thickness. No randomized Qhull joggle or parameter retry.
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree


def _source_ulp(values):
    if values.dtype.kind != "f":
        return 0.
    return float(abs(np.spacing(np.max(np.abs(values)))))


def coordinate_support_bounds(points, queries):
    raw, rq = np.asarray(points), np.asarray(queries)
    if (raw.ndim != 2 or raw.shape[1] != 3 or len(raw) == 0
            or rq.ndim != 2 or rq.shape[1] != 3 or len(rq) == 0
            or raw.dtype.kind not in "fiu" or rq.dtype.kind not in "fiu"
            or not np.isfinite(raw).all() or not np.isfinite(rq).all()):
        raise ValueError("support and queries must be nonempty finite Nx3 coordinates")
    p, q = raw.astype(np.float64), rq.astype(np.float64)
    center = p.mean(axis=0)
    centered, cq = p - center, q - center
    scale = max(1., float(np.max(np.abs(p))), float(np.max(np.abs(q))))
    arithmetic = 128 * np.finfo(np.float64).eps * scale
    tolerance = max(arithmetic * 4, 4 * np.sqrt(3) * max(_source_ulp(raw), _source_ulp(rq)))
    _, _, basis = np.linalg.svd(centered, full_matrices=True if len(p) < 3 else False)
    projected = centered @ basis.T
    keep = np.max(np.abs(projected), axis=0) > tolerance
    rank = int(keep.sum())
    axes = basis[keep]
    local = centered @ axes.T
    local_q = cq @ axes.T
    hull, hull_status = None, "AFFINE_POINT_OR_INTERVAL"
    directions = [*np.eye(3), *(-np.eye(3)), *basis, *(-basis)]
    if rank >= 2:
        try:
            hull = ConvexHull(local)
            # Only the strongest proposed face for each query is needed to
            # produce a useful certificate. This bounds the expensive ALL-point
            # support recheck by query count, not potentially 100k hull facets.
            # No support points are removed, and every chosen plane is rechecked.
            proposed = local_q @ hull.equations[:, :-1].T + hull.equations[:, -1]
            faces = np.unique(proposed.argmax(axis=1))
            directions.extend(hull.equations[faces, :-1] @ axes)
            hull_status = "CANDIDATES_FROM_QHULL_ORIGINAL_SUPPORT_RECHECKED"
        except QhullError:
            # This is a declared fallback to weaker certificates, not a new
            # geometry or a retry with relaxed/randomized Qhull options.
            hull_status = "QHULL_UNRESOLVED_FALLBACK_CERTIFICATES_ONLY"
    nearest_distance, nearest_indices = cKDTree(p).query(q, k=1, workers=1)
    directions.extend(cq)
    directions.extend(q - p[nearest_indices])
    null_q = cq - (local_q @ axes if rank else 0.)
    directions.extend(null_q)
    direction = np.asarray(directions)
    norms = np.linalg.norm(direction, axis=1)
    direction = direction[norms > 0] / norms[norms > 0, None]
    maxima = np.full(len(direction), -np.inf)
    argmax = np.zeros(len(direction), dtype=np.int64)
    for start in range(0, len(p), 1024):
        dots = centered[start:start + 1024] @ direction.T
        indices = dots.argmax(axis=0)
        values = dots[indices, np.arange(len(direction))]
        improved = values > maxima
        maxima[improved] = values[improved]; argmax[improved] = start + indices[improved]
    violations = cq @ direction.T - maxima
    selected = violations.argmax(axis=1)
    lower = np.maximum(0., violations[np.arange(len(q)), selected] - tolerance)
    upper, witnesses = [], []
    for i, target in enumerate(q):
        best = {"center_weight": 0., "point_indices": [int(nearest_indices[i])], "point_weights": [1.]}
        distance = float(nearest_distance[i])
        candidates = [{"center_weight": 1., "point_indices": [], "point_weights": []}]
        if rank == 1:
            lo, hi = int(local[:, 0].argmin()), int(local[:, 0].argmax())
            alpha = float(np.clip((local_q[i, 0] - local[lo, 0]) / (local[hi, 0] - local[lo, 0]), 0., 1.))
            candidates.append({"center_weight": 0., "point_indices": [lo, hi], "point_weights": [1-alpha, alpha]})
        elif hull is not None:
            # Fan from the convex center to the first boundary facet on the
            # query ray. Nonnegative clipped facet weights are re-evaluated in
            # original XYZ, so projection/rank errors only loosen the upper bound.
            denominators = hull.equations[:, :-1] @ local_q[i]
            useful = np.flatnonzero((denominators > 0) & (hull.equations[:, -1] < 0))
            if len(useful):
                factors = -hull.equations[useful, -1] / denominators[useful]
                first = float(factors.min())
                band = arithmetic / max(float(np.linalg.norm(local_q[i])), arithmetic)
                # A polygonal face may be triangulated into several coplanar
                # facets. Try every numerically tied triangle, not an arbitrary
                # first triangle that might not contain the ray intersection.
                for pick in np.flatnonzero(factors <= first + band):
                    facet, factor = useful[pick], float(factors[pick])
                    ids = hull.simplices[facet]
                    boundary = factor * local_q[i]
                    matrix = np.vstack((local[ids].T, np.ones(len(ids))))
                    weights = np.linalg.lstsq(matrix, np.r_[boundary, 1.], rcond=None)[0]
                    weights = np.maximum(weights, 0.)
                    if weights.sum() > 0 and factor > 0:
                        weights /= weights.sum()
                        boundary_weight = min(1., 1. / factor)
                        candidates.append({"center_weight": 1-boundary_weight, "point_indices": ids.tolist(),
                            "point_weights": (weights * boundary_weight).tolist()})
        for witness in candidates:
            reconstruction = witness["center_weight"] * center
            if witness["point_indices"]:
                reconstruction += np.asarray(witness["point_weights"]) @ p[witness["point_indices"]]
            residual = float(np.linalg.norm(target - reconstruction))
            if residual < distance:
                best, distance = witness, residual
        if (best["center_weight"] < 0 or min(best["point_weights"], default=0.) < 0
                or abs(best["center_weight"] + sum(best["point_weights"]) - 1.) > 1e-12):
            raise ValueError("invalid convex witness")
        upper.append(distance + arithmetic); witnesses.append(best)
    upper = np.asarray(upper)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower > upper + tolerance).any():
        raise ValueError("coordinate certificate numerical inconsistency")
    status = ["OUTSIDE_CERTIFIED" if l > tolerance else "WITHIN_TOLERANCE_WITNESS" if u <= tolerance
        else "UNRESOLVED" for l, u in zip(lower, upper)]
    return {"lower_bound_m": lower, "upper_bound_m": upper, "status": status,
        "numerical_bound_m": tolerance, "arithmetic_bound_m": arithmetic, "affine_rank": rank,
        "hull_status": hull_status, "support_point_count": len(p), "witnesses": witnesses,
        "candidate_direction_count": len(direction),
        "center_current_sensor_m": center, "source_coordinate_dtype": raw.dtype.str,
        "lower_certificate_directions": direction[selected],
        "lower_certificate_support_indices": argmax[selected]}
