"""Transparent non-learning baseline for outgoing directions from a range image.

This is an engineering baseline, not the project learning contribution.  It
uses only the current robot-frame range image and valid mask.  Graph, spline,
pose, world identity and future observations are deliberately absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _circular_runs(active: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(active, dtype=bool)
    if not np.any(values):
        return []
    if np.all(values):
        return [np.arange(len(values), dtype=np.int64)]
    first_inactive = int(np.flatnonzero(~values)[0])
    rolled = np.roll(values, -first_inactive)
    runs: list[np.ndarray] = []
    index = 0
    while index < len(rolled):
        if not rolled[index]:
            index += 1
            continue
        end = index
        while end < len(rolled) and rolled[end]:
            end += 1
        runs.append((np.arange(index, end) + first_inactive) % len(values))
        index = end
    return runs


def _circular_mean(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values.astype(np.float32, copy=True)
    if width % 2 == 0:
        raise ValueError("circular smoothing width must be odd")
    half = width // 2
    padded = np.concatenate((values[-half:], values, values[:half]))
    kernel = np.full(width, 1.0 / width, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def circular_distance_deg(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class RangeExitBaselineConfig:
    horizon_min_elevation_deg: float = -5.0
    horizon_max_elevation_deg: float = 5.0
    smoothing_columns: int = 15
    absolute_open_range_m: float = 7.0
    adaptive_percentile: float = 62.0
    minimum_sector_width_deg: float = 4.0
    maximum_exits: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RangeExitBaseline:
    """Extract contiguous long-range sectors as candidate tunnel exits."""

    def __init__(self, config: RangeExitBaselineConfig | None = None) -> None:
        self.config = config or RangeExitBaselineConfig()

    def predict(
        self,
        range_m: np.ndarray,
        valid_mask: np.ndarray,
        elevation_deg: np.ndarray,
    ) -> dict[str, Any]:
        ranges = np.asarray(range_m, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool)
        elevation = np.asarray(elevation_deg, dtype=np.float32)
        if ranges.ndim != 2 or valid.shape != ranges.shape:
            raise ValueError(f"range/valid shapes must match [rings, columns], got {ranges.shape}/{valid.shape}")
        if len(elevation) != ranges.shape[0]:
            raise ValueError("elevation count does not match range rows")
        rows = (elevation >= self.config.horizon_min_elevation_deg) & (
            elevation <= self.config.horizon_max_elevation_deg
        )
        if not np.any(rows):
            raise ValueError("configured horizon selects no LiDAR rows")

        selected = np.where(valid[rows], ranges[rows], 0.0)
        profile = np.max(selected, axis=0)
        smoothed = _circular_mean(profile, self.config.smoothing_columns)
        positive = smoothed[smoothed > 0]
        adaptive = (
            float(np.percentile(positive, self.config.adaptive_percentile))
            if positive.size
            else float("inf")
        )
        threshold = max(self.config.absolute_open_range_m, adaptive)
        active = smoothed >= threshold
        columns = ranges.shape[1]
        minimum_columns = max(
            1, int(np.ceil(self.config.minimum_sector_width_deg / (360.0 / columns)))
        )
        runs = [run for run in _circular_runs(active) if len(run) >= minimum_columns]
        runs.sort(key=lambda run: (-float(np.max(smoothed[run])), int(run[0])))
        runs = runs[: self.config.maximum_exits]

        headings = []
        sectors = []
        for run in runs:
            peak_column = int(run[int(np.argmax(smoothed[run]))])
            heading = float(peak_column * 360.0 / columns)
            headings.append(heading)
            sectors.append(
                {
                    "heading_robot_deg": heading,
                    "peak_range_m": float(smoothed[peak_column]),
                    "angular_width_deg": float(len(run) * 360.0 / columns),
                    "start_column": int(run[0]),
                    "end_column": int(run[-1]),
                }
            )
        headings.sort()
        return {
            "headings_robot_deg": headings,
            "branch_count": len(headings),
            "horizontal_range_profile_m": profile.astype(np.float32),
            "smoothed_range_profile_m": smoothed.astype(np.float32),
            "active_mask": active.astype(np.uint8),
            "threshold_m": float(threshold),
            "sectors": sectors,
            "config": self.config.to_dict(),
        }

