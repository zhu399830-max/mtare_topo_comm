"""Full frozen sensor context for proposals, observed10m surface for fitting."""
from dataclasses import replace
import numpy as np
from .primitive_relation_nonlearning import (
    _register_points, _candidate_runs, _candidates_from_masks, _circular_runs, _sector_mask, ELEVATION_DEG,
)
from .range_exit_baseline import RangeExitBaseline
from .observed_primitive_candidates import ObservedPrimitiveCandidate, observed_primitive_candidates
from .supported_primitive_fit import fit_supported_primitive


def full_sensor_local_fit(range_valid, translation, yaw, *, fit_config, exit_config, rotations=None):
    values = np.asarray(range_valid)
    translation, yaw = np.asarray(translation, float), np.asarray(yaw, float)
    if (values.shape != (5, 2, 16, 720) or not np.isfinite(values).all()
            or np.any(values[:, 0] < 0) or np.any(values[:, 0] > 1)
            or np.any((values[:, 1] != 0) & (values[:, 1] != 1))
            or np.any((values[:, 1] == 1) & (values[:, 0] <= 0))):
        raise ValueError('finite original normalized ranges and binary validity required')
    # Reuse strict configuration/motion validation without source or model reads.
    observed_primitive_candidates(np.empty((0, 3)), np.array([], int), translation, yaw,
                                 fit_config=fit_config, exit_config=exit_config)
    if rotations is not None:
        from .rigid_scan_registration import validate_rotations,register_returns,rotate_sector_mask
        rotations=validate_rotations(rotations)
    masks=np.zeros((5,720),bool);score=np.zeros(720)
    detector = RangeExitBaseline(replace(exit_config, maximum_exits=720))
    results = []; union = np.zeros(720, bool)
    for frame in range(5):
        result = detector.predict(values[frame, 0]*50., values[frame, 1], np.asarray(ELEVATION_DEG))
        if len(result['sectors']) > exit_config.maximum_exits:
            raise ValueError('per-frame capacity exceeded; no truncation')
        results.append(result)
        for sector in result['sectors']:
            if rotations is None:
                support=_sector_mask((sector['heading_robot_deg']+yaw[frame]) % 360,sector['angular_width_deg'])
            else:
                support=rotate_sector_mask(_sector_mask(sector['heading_robot_deg'],sector['angular_width_deg']),rotations[frame])
            union |= support;masks[frame] |= support
            score[support]=np.maximum(score[support],sector['peak_range_m'])
    if not np.any(union): return ()
    if len(_circular_runs(union)) > 32:
        raise ValueError('combined capacity exceeded; no truncation')

    class Saved:
        def __init__(self): self.i = 0
        def predict(self, *args):
            result = results[self.i]; self.i += 1; return result

    if rotations is None:
        candidates = _candidate_runs(values, yaw, Saved())
        points, _ = _register_points(values, translation, yaw)
        ids = np.flatnonzero(values[:, 1].reshape(-1))
    else:
        candidates=_candidates_from_masks(masks,score)
        points,ids=register_returns(values,translation,rotations)
    local = np.linalg.norm(points, axis=1) <= 10.
    points, ids = points[local], ids[local]
    return tuple(ObservedPrimitiveCandidate(c.heading_deg, c.angular_width_deg, c.bidirectional,
        tuple(int(x) for x in np.flatnonzero(c.temporal_visibility)),
        fit_supported_primitive(points, ids, heading_deg=c.heading_deg, angular_width_deg=c.angular_width_deg,
                                bidirectional=c.bidirectional, config=fit_config),
        tuple(int(x) for x in np.flatnonzero(masks[4]))) for c in candidates)
