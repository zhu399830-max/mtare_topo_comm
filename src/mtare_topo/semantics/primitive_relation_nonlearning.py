"""Same-input non-learning baseline for swept primitives and port relations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.optimize import minimize_scalar

from mtare_topo.data.cano_sensor_smoke import ELEVATION_DEG, lidar_local_directions
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline


HISTORY_FRAMES = 5
MAXIMUM_SLOTS = 32
MAXIMUM_RANGE_M = 50.0


def _circular_distance(first: np.ndarray | float, second: float) -> np.ndarray:
    return np.abs((np.asarray(first) - float(second) + 180.0) % 360.0 - 180.0)


def _circular_runs(mask: np.ndarray) -> list[np.ndarray]:
    active = np.asarray(mask, dtype=bool)
    if not np.any(active):
        return []
    if np.all(active):
        return [np.arange(len(active), dtype=np.int64)]
    first_inactive = int(np.flatnonzero(~active)[0])
    rolled = np.roll(active, -first_inactive)
    runs: list[np.ndarray] = []
    index = 0
    while index < len(rolled):
        if not rolled[index]:
            index += 1
            continue
        end = index
        while end < len(rolled) and rolled[end]:
            end += 1
        runs.append((np.arange(index, end) + first_inactive) % len(active))
        index = end
    return runs


@dataclass(frozen=True)
class PrimitiveRelationBaselineConfig:
    minimum_candidate_points: int = 64
    section_quantile: float = 0.05
    angular_surface_margin_deg: float = 15.0
    minimum_axis_extent_m: float = 1.0

    def __post_init__(self) -> None:
        if self.minimum_candidate_points < 16:
            raise ValueError("minimum candidate points must be at least 16")
        if not 0.0 < self.section_quantile < 0.25:
            raise ValueError("section quantile must lie in (0,0.25)")
        if not 0.0 <= self.angular_surface_margin_deg <= 45.0:
            raise ValueError("angular surface margin must lie in [0,45]")
        if self.minimum_axis_extent_m <= 0.0:
            raise ValueError("minimum axis extent must be positive")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class NonlearningPrimitiveRelationPrediction:
    primitive_mask: np.ndarray
    axis_control_current_sensor_m: np.ndarray
    endpoint_half_axes_m: np.ndarray
    endpoint_shape_exponent: np.ndarray
    geometry_uncertainty: np.ndarray
    temporal_visibility: np.ndarray
    temporal_destination: np.ndarray
    endpoint_attachment: np.ndarray
    disconnected_overlap: np.ndarray
    azimuth_support: np.ndarray

    def __post_init__(self) -> None:
        expected = {
            "primitive_mask": (32,),
            "axis_control_current_sensor_m": (32, 3, 3),
            "endpoint_half_axes_m": (32, 2, 2),
            "endpoint_shape_exponent": (32, 2),
            "geometry_uncertainty": (32,),
            "temporal_visibility": (5, 32),
            "temporal_destination": (5, 32),
            "endpoint_attachment": (32, 2, 32, 2),
            "disconnected_overlap": (32, 32),
            "azimuth_support": (32, 720),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"non-learning output shape drift: {name}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"non-learning output is non-finite: {name}")
        mask = np.asarray(self.primitive_mask, dtype=bool)
        if not np.any(mask) or np.any(self.primitive_mask > 1):
            raise ValueError("non-learning baseline must return 1--32 binary slots")
        if np.any(self.endpoint_half_axes_m[mask] <= 0.0):
            raise ValueError("active non-learning half axes must be positive")
        if np.any((self.endpoint_shape_exponent[mask] < 2.0) | (self.endpoint_shape_exponent[mask] > 10.0)):
            raise ValueError("active non-learning exponents must lie in [2,10]")
        if not np.array_equal(self.endpoint_attachment, self.endpoint_attachment.transpose(2, 3, 0, 1)):
            raise ValueError("non-learning attachment must be symmetric")
        if not np.array_equal(self.disconnected_overlap, self.disconnected_overlap.T):
            raise ValueError("non-learning overlap must be symmetric")


@dataclass(frozen=True)
class _Candidate:
    heading_deg: float
    angular_width_deg: float
    support: np.ndarray
    temporal_visibility: np.ndarray
    bidirectional: bool


def _register_points(
    range_valid: np.ndarray,
    translation: np.ndarray,
    yaw_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    directions = np.asarray(lidar_local_directions(), dtype=np.float64)
    points: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    for frame in range(HISTORY_FRAMES):
        ranges = range_valid[frame, 0].astype(np.float64) * MAXIMUM_RANGE_M
        valid = range_valid[frame, 1].astype(bool)
        local = directions[valid] * ranges[valid, None]
        angle = math.radians(float(yaw_deg[frame]))
        cosine, sine = math.cos(angle), math.sin(angle)
        rotated = local.copy()
        rotated[:, 0] = cosine * local[:, 0] - sine * local[:, 1]
        rotated[:, 1] = sine * local[:, 0] + cosine * local[:, 1]
        points.append(rotated + translation[frame])
        frames.append(np.full(len(local), frame, dtype=np.int16))
    if not points or sum(len(value) for value in points) == 0:
        raise RuntimeError("non-learning baseline received no finite LiDAR return")
    return np.concatenate(points), np.concatenate(frames)


def _sector_mask(heading_deg: float, width_deg: float) -> np.ndarray:
    columns = np.arange(720, dtype=np.float64) * 0.5
    return _circular_distance(columns, heading_deg) <= max(2.0, 0.5 * width_deg)


def _candidate_runs(
    range_valid: np.ndarray,
    yaw_deg: np.ndarray,
    exit_baseline: RangeExitBaseline,
) -> list[_Candidate]:
    frame_masks = np.zeros((HISTORY_FRAMES, 720), dtype=bool)
    score = np.zeros(720, dtype=np.float64)
    for frame in range(HISTORY_FRAMES):
        result = exit_baseline.predict(
            range_valid[frame, 0] * MAXIMUM_RANGE_M,
            range_valid[frame, 1],
            np.asarray(ELEVATION_DEG),
        )
        for sector in result["sectors"]:
            heading = (float(sector["heading_robot_deg"]) + float(yaw_deg[frame])) % 360.0
            support = _sector_mask(heading, float(sector["angular_width_deg"]))
            frame_masks[frame] |= support
            score[support] = np.maximum(score[support], float(sector["peak_range_m"]))
    runs = _circular_runs(np.any(frame_masks, axis=0))
    if not runs:
        # A fully enclosed/short observation still needs a deterministic local
        # axis baseline.  Use the strongest current horizontal column.
        profile = np.max(range_valid[-1, 0] * range_valid[-1, 1], axis=0)
        peak = int(np.argmax(profile)); support = _sector_mask(peak * 0.5, 4.0)
        frame_masks[:, support] = True; score[support] = profile[peak] * MAXIMUM_RANGE_M
        runs = [np.flatnonzero(support)]

    raw: list[_Candidate] = []
    for run in runs:
        weights = np.maximum(score[run], 1e-6)
        angle = np.radians(run.astype(np.float64) * 0.5)
        heading = math.degrees(math.atan2(float(np.sum(weights * np.sin(angle))), float(np.sum(weights * np.cos(angle))))) % 360.0
        support = np.zeros(720, dtype=bool); support[run] = True
        temporal = np.any(frame_masks[:, support], axis=1)
        raw.append(_Candidate(heading, len(run) * 0.5, support, temporal, False))

    if len(raw) == 2 and abs(float(_circular_distance(raw[0].heading_deg, raw[1].heading_deg)) - 180.0) <= max(raw[0].angular_width_deg, raw[1].angular_width_deg, 12.0):
        first, second = raw
        doubled = np.radians(np.asarray((first.heading_deg, second.heading_deg)) * 2.0)
        heading = 0.5 * math.degrees(math.atan2(float(np.sin(doubled).sum()), float(np.cos(doubled).sum()))) % 180.0
        return [_Candidate(
            heading, max(first.angular_width_deg, second.angular_width_deg),
            first.support | second.support,
            first.temporal_visibility | second.temporal_visibility, True,
        )]
    raw.sort(key=lambda value: value.heading_deg)
    return raw[:MAXIMUM_SLOTS]


def _fit_exponent(lateral: np.ndarray, vertical: np.ndarray, half_axes: np.ndarray) -> tuple[float, float]:
    normalized_lateral = np.clip(np.abs(lateral) / max(float(half_axes[0]), 1e-4), 0.0, 4.0)
    normalized_vertical = np.clip(np.abs(vertical) / max(float(half_axes[1]), 1e-4), 0.0, 4.0)

    def objective(exponent: float) -> float:
        residual = np.abs(normalized_lateral**exponent + normalized_vertical**exponent - 1.0)
        return float(np.median(np.minimum(residual, 4.0)))

    result = minimize_scalar(objective, bounds=(2.0, 10.0), method="bounded", options={"xatol": 1e-3})
    exponent = float(np.clip(result.x if result.success else 2.0, 2.0, 10.0))
    return exponent, objective(exponent)


def _fit_candidate(
    candidate: _Candidate,
    points: np.ndarray,
    config: PrimitiveRelationBaselineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    angle = math.radians(candidate.heading_deg)
    axis_xy = np.asarray((math.cos(angle), math.sin(angle)))
    side_xy = np.asarray((-axis_xy[1], axis_xy[0]))
    longitudinal = points[:, :2] @ axis_xy
    lateral = points[:, :2] @ side_xy
    point_heading = np.degrees(np.arctan2(points[:, 1], points[:, 0])) % 360.0
    directional = np.minimum(
        _circular_distance(point_heading, candidate.heading_deg),
        _circular_distance(point_heading, candidate.heading_deg + 180.0),
    ) <= 0.5 * candidate.angular_width_deg + config.angular_surface_margin_deg
    if not candidate.bidirectional:
        directional &= longitudinal > 0.0
    selected = np.flatnonzero(directional)
    if len(selected) < config.minimum_candidate_points:
        selected = np.argsort(np.minimum(
            _circular_distance(point_heading, candidate.heading_deg),
            _circular_distance(point_heading, candidate.heading_deg + 180.0),
        ))[: min(len(points), config.minimum_candidate_points)]
    s = longitudinal[selected]; y = lateral[selected]; z = points[selected, 2]
    low, high = np.quantile(s, (config.section_quantile, 1.0 - config.section_quantile))
    if not candidate.bidirectional:
        low = max(0.0, float(low))
    if high - low < config.minimum_axis_extent_m:
        high = low + config.minimum_axis_extent_m
    section_s = np.linspace(float(low), float(high), 3)
    span = max((high - low) / 2.0, config.minimum_axis_extent_m)
    controls = np.zeros((3, 3), dtype=np.float64)
    section_axes = np.zeros((3, 2), dtype=np.float64)
    section_values: list[tuple[np.ndarray, np.ndarray]] = []
    q = config.section_quantile
    for index, center_s in enumerate(section_s):
        inside = np.abs(s - center_s) <= span
        if int(inside.sum()) < 16:
            nearest = np.argsort(np.abs(s - center_s))[: min(len(s), max(16, len(s) // 3))]
            inside = np.zeros(len(s), dtype=bool); inside[nearest] = True
        side_bounds = np.quantile(y[inside], (q, 1.0 - q))
        vertical_bounds = np.quantile(z[inside], (q, 1.0 - q))
        lateral_center = float(side_bounds.mean()); vertical_center = float(vertical_bounds.mean())
        controls[index, :2] = center_s * axis_xy + lateral_center * side_xy
        controls[index, 2] = vertical_center
        section_axes[index] = np.maximum(
            (0.5 * float(np.diff(side_bounds)[0]), 0.5 * float(np.diff(vertical_bounds)[0])),
            0.1,
        )
        section_values.append((y[inside] - lateral_center, z[inside] - vertical_center))
    half_axes = section_axes[[0, 2]]
    exponents = np.zeros(2, dtype=np.float64); residuals = []
    for endpoint, section_index in enumerate((0, 2)):
        exponents[endpoint], residual = _fit_exponent(
            *section_values[section_index], half_axes[endpoint],
        )
        residuals.append(residual)
    return controls, half_axes, exponents, float(np.mean(residuals))


class RobustPrimitiveRelationBaseline:
    """Fit visible tunnel stubs and their relations without learned weights."""

    def __init__(
        self,
        config: PrimitiveRelationBaselineConfig | None = None,
        exit_baseline: RangeExitBaseline | None = None,
    ) -> None:
        self.config = config or PrimitiveRelationBaselineConfig()
        self.exit_baseline = exit_baseline or RangeExitBaseline()

    def predict(
        self,
        range_valid: np.ndarray,
        relative_translation_current_sensor_m: np.ndarray,
        relative_yaw_current_sensor_deg: np.ndarray,
    ) -> NonlearningPrimitiveRelationPrediction:
        values = np.asarray(range_valid, dtype=np.float32)
        translation = np.asarray(relative_translation_current_sensor_m, dtype=np.float32)
        yaw = np.asarray(relative_yaw_current_sensor_deg, dtype=np.float32)
        if values.shape != (5, 2, 16, 720) or translation.shape != (5, 3) or yaw.shape != (5,):
            raise ValueError("non-learning baseline expects [5,2,16,720], [5,3], [5]")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(translation)) or not np.all(np.isfinite(yaw)):
            raise ValueError("non-learning student input must be finite")
        if np.any((values[:, 0] < 0.0) | (values[:, 0] > 1.0)) or np.any((values[:, 1] != 0.0) & (values[:, 1] != 1.0)):
            raise ValueError("non-learning range/valid contract drift")
        if not np.array_equal(translation[-1], np.zeros(3, dtype=np.float32)) or yaw[-1] != 0.0:
            raise ValueError("non-learning current relative odometry must be zero")

        points, _ = _register_points(values, translation, yaw)
        candidates = _candidate_runs(values, yaw, self.exit_baseline)
        count = len(candidates)
        mask = np.zeros(32, dtype=np.uint8); mask[:count] = 1
        axes = np.zeros((32, 3, 3), dtype=np.float32)
        half_axes = np.zeros((32, 2, 2), dtype=np.float32)
        exponent = np.zeros((32, 2), dtype=np.float32)
        uncertainty = np.zeros(32, dtype=np.float32)
        temporal = np.zeros((5, 32), dtype=np.uint8)
        destination = np.full((5, 32), 32, dtype=np.int8)
        support = np.zeros((32, 720), dtype=np.uint8)
        for slot, candidate in enumerate(candidates):
            controls, sizes, shapes, residual = _fit_candidate(candidate, points, self.config)
            axes[slot] = controls; half_axes[slot] = sizes; exponent[slot] = shapes
            uncertainty[slot] = residual; temporal[:, slot] = candidate.temporal_visibility
            destination[candidate.temporal_visibility, slot] = slot
            support[slot] = candidate.support

        attachment = np.zeros((32, 2, 32, 2), dtype=np.uint8)
        overlap = np.zeros((32, 32), dtype=np.uint8)
        for first in range(count):
            for second in range(first + 1, count):
                endpoint_distance = np.linalg.norm(
                    axes[first, (0, 2), None] - axes[second, None, (0, 2)], axis=-1,
                )
                endpoint_pair = np.unravel_index(int(np.argmin(endpoint_distance)), (2, 2))
                radius = 0.5 * (
                    float(np.mean(half_axes[first, endpoint_pair[0]]))
                    + float(np.mean(half_axes[second, endpoint_pair[1]]))
                )
                both_ego_stubs = (
                    not candidates[first].bidirectional
                    and not candidates[second].bidirectional
                )
                if both_ego_stubs:
                    endpoint_pair = (
                        int(np.argmin(np.linalg.norm(axes[first, (0, 2)], axis=1))),
                        int(np.argmin(np.linalg.norm(axes[second, (0, 2)], axis=1))),
                    )
                if both_ego_stubs or float(endpoint_distance[endpoint_pair]) <= max(0.5, radius):
                    first_endpoint, second_endpoint = endpoint_pair
                    attachment[first, first_endpoint, second, second_endpoint] = 1
                    attachment[second, second_endpoint, first, first_endpoint] = 1
                elif np.any(support[first].astype(bool) & support[second].astype(bool)):
                    overlap[first, second] = overlap[second, first] = 1
        return NonlearningPrimitiveRelationPrediction(
            mask, axes, half_axes, exponent, uncertainty, temporal, destination,
            attachment, overlap, support,
        )


__all__ = [
    "NonlearningPrimitiveRelationPrediction", "PrimitiveRelationBaselineConfig",
    "RobustPrimitiveRelationBaseline",
]
