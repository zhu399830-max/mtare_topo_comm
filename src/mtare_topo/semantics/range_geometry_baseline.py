"""Transparent PCA/section baseline for local tunnel geometry from one LiDAR scan."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from mtare_topo.data.cano_sensor_smoke import AZIMUTH_COLUMNS, ELEVATION_DEG, MAX_RANGE_M, NEAR_RANGE_M


@dataclass(frozen=True)
class RangeGeometryBaselineConfig:
    local_range_m: float = 20.0
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    longitudinal_bins: int = 5
    minimum_points: int = 128
    minimum_points_per_bin: int = 20

    def __post_init__(self) -> None:
        if not NEAR_RANGE_M < self.local_range_m <= MAX_RANGE_M:
            raise ValueError("local_range_m must lie inside the sensor range")
        if not 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0:
            raise ValueError("invalid robust section quantiles")
        if self.longitudinal_bins < 3 or self.minimum_points < 1 or self.minimum_points_per_bin < 3:
            raise ValueError("invalid geometry baseline sample contract")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RangeGeometryBaseline:
    """Estimate axis, cross-section, slope and curvature without learned weights.

    The largest horizontal PCA direction gives an unoriented tunnel axis.
    Equal-occupancy longitudinal sections then provide robust left/right and
    floor/ceiling envelopes. A line through section-height centres estimates
    slope; a quadratic through lateral centres estimates planar curvature.
    """

    def __init__(self, config: RangeGeometryBaselineConfig | None = None) -> None:
        self.config = config or RangeGeometryBaselineConfig()

    def _points(self, range_m: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        ranges = np.asarray(range_m, dtype=np.float64)
        valid = np.asarray(valid_mask, dtype=np.bool_)
        if ranges.shape != (len(ELEVATION_DEG), AZIMUTH_COLUMNS) or valid.shape != ranges.shape:
            raise ValueError("range geometry baseline expects one [16,720] scan and mask")
        if not np.all(np.isfinite(ranges)) or np.any(ranges < NEAR_RANGE_M) or np.any(ranges > MAX_RANGE_M):
            raise ValueError("range scan violates the finite sensor contract")
        selected = valid & (ranges <= self.config.local_range_m)
        if int(selected.sum()) < self.config.minimum_points:
            raise RuntimeError("insufficient local returns for non-learning geometry estimation")
        elevation, azimuth = np.meshgrid(
            np.radians(ELEVATION_DEG),
            np.arange(AZIMUTH_COLUMNS, dtype=np.float64) * (2.0 * np.pi / AZIMUTH_COLUMNS),
            indexing="ij",
        )
        cosine = np.cos(elevation)
        xyz = np.stack(
            (
                ranges * cosine * np.cos(azimuth),
                ranges * cosine * np.sin(azimuth),
                ranges * np.sin(elevation),
            ),
            axis=-1,
        )
        return xyz[selected]

    def predict(self, range_m: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
        points = self._points(range_m, valid_mask)
        horizontal = points[:, :2] - np.median(points[:, :2], axis=0)
        covariance = horizontal.T @ horizontal / len(horizontal)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis_xy = eigenvectors[:, int(np.argmax(eigenvalues))]
        if axis_xy[0] < 0.0 or (abs(axis_xy[0]) <= 1e-12 and axis_xy[1] < 0.0):
            axis_xy = -axis_xy
        lateral_xy = np.asarray((-axis_xy[1], axis_xy[0]))
        longitudinal = points[:, :2] @ axis_xy
        lateral = points[:, :2] @ lateral_xy

        boundaries = np.quantile(longitudinal, np.linspace(0.0, 1.0, self.config.longitudinal_bins + 1))
        section_arc: list[float] = []
        lateral_centres: list[float] = []
        vertical_centres: list[float] = []
        widths: list[float] = []
        heights: list[float] = []
        for index in range(self.config.longitudinal_bins):
            inside = (longitudinal >= boundaries[index]) & (
                longitudinal <= boundaries[index + 1]
                if index == self.config.longitudinal_bins - 1
                else longitudinal < boundaries[index + 1]
            )
            if int(inside.sum()) < self.config.minimum_points_per_bin:
                continue
            side = np.quantile(lateral[inside], (self.config.lower_quantile, self.config.upper_quantile))
            vertical = np.quantile(points[inside, 2], (self.config.lower_quantile, self.config.upper_quantile))
            section_arc.append(float(np.median(longitudinal[inside])))
            lateral_centres.append(float(np.mean(side)))
            vertical_centres.append(float(np.mean(vertical)))
            widths.append(float(side[1] - side[0]))
            heights.append(float(vertical[1] - vertical[0]))
        if len(section_arc) < 3:
            raise RuntimeError("insufficient populated longitudinal sections")
        arc = np.asarray(section_arc, dtype=np.float64)
        centered_arc = arc - float(np.mean(arc))
        vertical_coefficients = np.polyfit(centered_arc, np.asarray(vertical_centres), deg=1)
        lateral_coefficients = np.polyfit(centered_arc, np.asarray(lateral_centres), deg=2)
        slope = float(vertical_coefficients[0])
        first_derivative = float(lateral_coefficients[1])
        second_derivative = float(2.0 * lateral_coefficients[0])
        curvature = abs(second_derivative) / (1.0 + first_derivative * first_derivative) ** 1.5
        axis = np.asarray((axis_xy[0], axis_xy[1], slope), dtype=np.float64)
        axis /= np.linalg.norm(axis)
        return {
            "local_axis": axis.tolist(),
            "width_m": float(np.median(widths)),
            "height_m": float(
                np.diff(
                    np.quantile(
                        points[:, 2],
                        (self.config.lower_quantile, self.config.upper_quantile),
                    )
                )[0]
            ),
            "slope_deg": float(math.degrees(math.atan(slope))),
            "curvature_per_m": float(curvature),
            "local_return_count": int(len(points)),
            "populated_sections": len(section_arc),
            "config": self.config.to_dict(),
        }


__all__ = ["RangeGeometryBaseline", "RangeGeometryBaselineConfig"]
