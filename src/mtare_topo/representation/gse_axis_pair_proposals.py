"""Unfiltered chord-pair hypotheses, not junctions, openings or graph edges.

Accepts geometry only, not teacher membership or old attachment heuristics.
Chord approximation can be poor on curves; expose its residual. Parallel or
degenerate pairs have no unique position. Skew-line midpoints remain untrusted
hypotheses with their full 3D separation, never asserted connections.
"""
import itertools
import numpy as np


def axis_pair_proposals(axes_m, valid, *, coordinate_frame):
    if (not isinstance(axes_m, np.ndarray) or axes_m.ndim != 3
            or axes_m.shape[1:] != (3, 3) or len(axes_m) > 32
            or axes_m.dtype not in (np.dtype('float32'), np.dtype('float64'))):
        raise ValueError('at most32 floating three-control axes required')
    if (not isinstance(valid, np.ndarray) or valid.dtype != np.bool_
            or valid.shape != (len(axes_m),)):
        raise ValueError('explicit boolean axis validity required')
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise ValueError('explicit coordinate frame required')
    if not np.isfinite(axes_m[valid]).all():
        raise ValueError('nonfinite supported geometry')
    axes = axes_m.astype(np.float64)
    # Numerical conditioning guard, not a fitted geometric acceptance radius.
    tolerance = 64 * np.finfo(axes_m.dtype).eps
    rows = []
    for i, j in itertools.combinations(np.flatnonzero(valid).tolist(), 2):
        a, b = axes[i], axes[j]
        da, db = a[2]-a[0], b[2]-b[0]
        la, lb = np.linalg.norm(da), np.linalg.norm(db)
        row = dict(axis_indices=[i, j], position_m=None, status='DEGENERATE',
                   connection_confirmed=False)
        if min(la, lb) <= tolerance:
            rows.append(row)
            continue
        u, v = da/la, db/lb
        design = np.column_stack((u, -v))
        singular = np.linalg.svd(design, compute_uv=False)
        row['absolute_direction_cosine'] = float(abs(np.dot(u, v)))
        row['singular_value_ratio'] = float(singular[-1]/singular[0])
        if singular[-1] <= tolerance*singular[0]:
            row['status'] = 'PARALLEL_POSITION_UNKNOWN'
            rows.append(row)
            continue
        t, s = np.linalg.lstsq(design, b[0]-a[0], rcond=None)[0]
        p, q = a[0]+t*u, b[0]+s*v
        midpoint = p+(q-p)/2
        deviation = []
        for control, direction in ((a, u), (b, v)):
            delta = control[1]-control[0]
            deviation.append(float(np.linalg.norm(delta-np.dot(delta, direction)*direction)))
        values = np.concatenate((midpoint, p, q, [t, s, la, lb]))
        if not np.isfinite(values).all():
            raise ValueError('nonfinite pair solution')
        row.update(status='UNVERIFIED_PAIR_HYPOTHESIS', position_m=midpoint.tolist(),
                   closest_points_m=[p.tolist(), q.tolist()],
                   separation_m=float(np.linalg.norm(p-q)),
                   extrapolation_m=[float(max(-t, t-la, 0.)), float(max(-s, s-lb, 0.))],
                   chord_deviation_m=deviation)
        rows.append(row)
    return dict(schema='gse_axis_pair_proposals_v1', coordinate_frame=coordinate_frame,
                valid_axis_count=int(valid.sum()), pair_count=len(rows), pairs=rows,
                learned=False, scientific_gate_pass=False,
                limitation='No terminal detector; chord pairs do not certify a junction. '
                           'Upstream axes must be observation-derived, not teacher-selected.')
