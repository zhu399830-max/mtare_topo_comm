"""Parameter/cropping vs spatial-layout diagnosis of three axis controls.

Both inputs are already aligned by a frozen scoring-only correspondence.
The two-segment interpolant is NOT the original map spline. Direction validity
uses input-quantization resolution, not a tuned geometric acceptance threshold.
Midpoint quadrature uses exact distances to finite line segments with a
Lipschitz error bound; short overlapping predictions cannot hide missing extent
because both directed distances and lengths are returned.
"""
import numpy as np


def _length(axis):
    return float(np.linalg.norm(np.diff(axis, axis=0), axis=1).sum())


def _chord_resolution(raw):
    if np.asarray(raw).dtype.kind != "f":
        return 0.0
    # Each endpoint coordinate can carry one source ULP of uncertainty.
    magnitude = np.max(np.abs(np.asarray(raw)))
    return float(2 * np.sqrt(3) * abs(np.spacing(magnitude)))


def _midpoints(axis, count):
    lengths = np.linalg.norm(np.diff(axis, axis=0), axis=1)
    total = lengths.sum()
    if total == 0:
        return np.repeat(axis[:1], count, axis=0)
    arc = (np.arange(count) + .5) * total / count
    segment = (arc >= lengths[0]).astype(int)
    fraction = (arc - np.where(segment == 0, 0, lengths[0])) / lengths[segment]
    return axis[segment] + fraction[:, None] * (axis[segment + 1] - axis[segment])


def _point_polyline_distance(points, axis):
    directions = np.diff(axis, axis=0)
    square = (directions * directions).sum(axis=1)
    delta = points[:, None] - axis[None, :2]
    numerator = (delta * directions).sum(axis=2)
    fraction = np.divide(numerator, square, out=np.zeros_like(numerator), where=square > 0)
    residual = delta - np.clip(fraction, 0, 1)[..., None] * directions
    return np.linalg.norm(residual, axis=2).min(axis=1)


def decompose_axes(predicted, teacher, samples=256):
    if type(samples) is not int or samples < 1:
        raise ValueError("positive integer quadrature count required")
    raw_pred, raw_teacher = np.asarray(predicted), np.asarray(teacher)
    pred, truth = raw_pred.astype(np.float64), raw_teacher.astype(np.float64)
    if pred.shape != (3, 3) or truth.shape != (3, 3) or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("axes must be finite 3x3 control points")
    delta = pred - truth
    lp, lt = _length(pred), _length(truth)
    pc, tc = pred[-1] - pred[0], truth[-1] - truth[0]
    pn, tn = np.linalg.norm(pc), np.linalg.norm(tc)
    pv, tv = bool(pn > _chord_resolution(raw_pred)), bool(tn > _chord_resolution(raw_teacher))
    result = {"point_mean_m": float(np.linalg.norm(delta, axis=1).mean()),
        "coordinate_mae_m": float(np.abs(delta).mean()),
        "total_rms_m": float(np.sqrt(np.square(delta).sum(axis=1).mean())),
        "axial_rms_m": None, "transverse_rms_m": None,
        "direction_error_deg": None, "undirected_direction_error_deg": None,
        "teacher_direction_resolved": tv, "prediction_direction_resolved": pv,
        "teacher_length_m": lt, "predicted_length_m": lp,
        "length_ratio": lp / lt if lt > 0 else None,
        "length_absolute_error_m": abs(lp - lt),
        "sensor_vertical_rms_m": float(np.sqrt(np.square(delta[:, 2]).mean())),
        "pred_to_teacher_polyline_m": float(_point_polyline_distance(_midpoints(pred, samples), truth).mean()),
        "teacher_to_pred_polyline_m": float(_point_polyline_distance(_midpoints(truth, samples), pred).mean()),
        "quadrature_samples_per_direction": samples, "quadrature_error_bound_m": (lp + lt) / (4 * samples)}
    result["polyline_symmetric_m"] = .5 * (result["pred_to_teacher_polyline_m"] + result["teacher_to_pred_polyline_m"])
    if tv:
        along = delta @ (tc / tn)
        normal = delta - along[:, None] * (tc / tn)
        result["axial_rms_m"] = float(np.sqrt(np.square(along).mean()))
        result["transverse_rms_m"] = float(np.sqrt(np.square(normal).sum(axis=1).mean()))
        result["axial_signed_mean_m"] = float(along.mean())
        result["axial_squared_error_sum_m2"] = float(np.square(along).sum())
        result["transverse_squared_error_sum_m2"] = float(np.square(normal).sum())
    else:
        result.update(dict.fromkeys(("axial_signed_mean_m", "axial_squared_error_sum_m2", "transverse_squared_error_sum_m2")))
    if pv and tv:
        cosine = float(np.clip(np.dot(pc / pn, tc / tn), -1., 1.))
        result["direction_error_deg"] = float(np.degrees(np.arccos(cosine)))
        result["undirected_direction_error_deg"] = float(np.degrees(np.arccos(abs(cosine))))
    return result
