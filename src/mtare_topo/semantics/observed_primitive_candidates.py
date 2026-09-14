"""Observation-only proposal and supported-fit adapter for compact LiDAR.

Modified historical baseline: retain its sector/axis heuristics, but never force
an empty candidate or silently truncate. Not a 3-D opening detector: horizontal
sector pooling can conflate height layers. No teacher, node or edge identity.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .primitive_relation_nonlearning import (
    HISTORY_FRAMES, MAXIMUM_RANGE_M, MAXIMUM_SLOTS, ELEVATION_DEG,
    PrimitiveRelationBaselineConfig, _candidate_runs, _circular_runs, _sector_mask,
)
from .range_exit_baseline import RangeExitBaseline, RangeExitBaselineConfig
from .supported_primitive_fit import SupportedPrimitiveFit, fit_supported_primitive


@dataclass(frozen=True)
class ObservedPrimitiveCandidate:
    heading_deg: float
    angular_width_deg: float
    bidirectional: bool
    contributing_frames: tuple[int, ...]
    fit: SupportedPrimitiveFit
    # Actual current-frame direction sectors, not fitted surface-ray membership.
    # None marks legacy producers without this evidence.
    current_sector_columns: tuple[int, ...] | None = None


def observed_primitive_candidates(
    points_xyz_m, source_flat_ray_index, relative_translation_current_sensor_m,
    relative_yaw_current_sensor_deg, *, fit_config: PrimitiveRelationBaselineConfig,
    exit_config: RangeExitBaselineConfig,
) -> tuple[ObservedPrimitiveCandidate, ...]:
    """Use exactly supplied ROI returns; do not clip ranges to invent returns.

    Translation/yaw describe past sensor frames in current sensor coordinates.
    Source IDs are frame*11520 + ring*720 + column. Recover original range from
    aligned point minus sensor origin, never from distance to the current origin.
    Caller authenticates points, motion, source IDs and ROI before this adapter.
    """
    p = np.asarray(points_xyz_m, dtype=float)
    ids = np.asarray(source_flat_ray_index)
    translation = np.asarray(relative_translation_current_sensor_m, dtype=float)
    yaw = np.asarray(relative_yaw_current_sensor_deg, dtype=float)
    if p.ndim != 2 or p.shape[1:] != (3,) or not np.isfinite(p).all():
        raise ValueError('finite aligned Nx3 points required')
    if (ids.shape != (len(p),) or ids.dtype.kind not in 'iu' or np.any(ids < 0)
            or np.any(ids >= 57600) or len(np.unique(ids)) != len(ids)):
        raise ValueError('unique original ray indices required')
    if (translation.shape != (5, 3) or yaw.shape != (5,)
            or not np.isfinite(translation).all() or not np.isfinite(yaw).all()
            or np.any(translation[-1] != 0) or yaw[-1] != 0):
        raise ValueError('finite causal motion with zero current transform required')
    if (not isinstance(exit_config, RangeExitBaselineConfig)
            or not np.isfinite(list(exit_config.to_dict().values())).all()
            or not isinstance(exit_config.maximum_exits, int)
            or not 1 <= exit_config.maximum_exits <= 720):
        raise ValueError('finite exit config and integer capacity required')
    if (exit_config.smoothing_columns < 1 or exit_config.smoothing_columns > 719
            or exit_config.smoothing_columns % 2 != 1
            or not 0 <= exit_config.adaptive_percentile <= 100
            or exit_config.absolute_open_range_m <= 0
            or not 0 < exit_config.minimum_sector_width_deg <= 360
            or exit_config.horizon_min_elevation_deg > exit_config.horizon_max_elevation_deg):
        raise ValueError('invalid sector configuration')
    frames = ids // 11520
    ranges = np.linalg.norm(p - translation[frames], axis=1)
    if np.any(ranges <= 0) or np.any(ranges > MAXIMUM_RANGE_M):
        raise ValueError('source ranges must lie in (0,50] metres')
    range_valid = np.zeros((5, 2, 16, 720), dtype=np.float32)
    ring, column = (ids % 11520)//720, ids % 720
    range_valid[frames, 0, ring, column] = ranges / MAXIMUM_RANGE_M
    range_valid[frames, 1, ring, column] = 1
    # Extract all sectors first, then explicitly enforce the declared old cap.
    detector = RangeExitBaseline(replace(exit_config, maximum_exits=720))
    results = []
    union = np.zeros(720, bool)
    for frame in range(HISTORY_FRAMES):
        result = detector.predict(range_valid[frame, 0]*MAXIMUM_RANGE_M,
                                  range_valid[frame, 1], np.asarray(ELEVATION_DEG))
        if len(result['sectors']) > exit_config.maximum_exits:
            raise ValueError('per-frame sector capacity exceeded; no truncation')
        results.append(result)
        for sector in result['sectors']:
            union |= _sector_mask((sector['heading_robot_deg'] + yaw[frame]) % 360,
                                  sector['angular_width_deg'])
    if not np.any(union):
        return ()  # bypass historical strongest-column forced candidate
    if len(_circular_runs(union)) > MAXIMUM_SLOTS:
        raise ValueError('combined candidate capacity exceeded; no truncation')

    class SavedSectorResults:
        def __init__(self):
            self.index = 0

        def predict(self, *args):
            result = results[self.index]
            self.index += 1
            return result

    candidates = _candidate_runs(range_valid, yaw, SavedSectorResults())
    return tuple(ObservedPrimitiveCandidate(
        c.heading_deg, c.angular_width_deg, c.bidirectional,
        tuple(int(f) for f in np.flatnonzero(c.temporal_visibility)),
        fit_supported_primitive(p, ids, heading_deg=c.heading_deg,
                                angular_width_deg=c.angular_width_deg,
                                bidirectional=c.bidirectional, config=fit_config),
    ) for c in candidates)
