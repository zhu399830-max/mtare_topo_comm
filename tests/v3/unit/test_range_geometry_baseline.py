from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.data.cano_sensor_smoke import AZIMUTH_COLUMNS, ELEVATION_DEG, MAX_RANGE_M, NEAR_RANGE_M
from mtare_topo.semantics.range_geometry_baseline import RangeGeometryBaseline


def _rectangular_tunnel_scan() -> tuple[np.ndarray, np.ndarray]:
    elevation, azimuth = np.meshgrid(
        np.radians(ELEVATION_DEG),
        np.arange(AZIMUTH_COLUMNS) * 2.0 * np.pi / AZIMUTH_COLUMNS,
        indexing="ij",
    )
    directions = np.stack(
        (
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ),
        axis=-1,
    )
    candidates = np.full((*directions.shape[:2], 4), np.inf)
    for index, (axis, plane) in enumerate(((1, 2.5), (1, -2.5), (2, 3.0), (2, -1.0))):
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = plane / directions[..., axis]
        candidates[..., index] = np.where(distance > 0.0, distance, np.inf)
    distance = np.min(candidates, axis=-1)
    valid = np.isfinite(distance) & (distance <= MAX_RANGE_M)
    ranges = np.where(valid, distance, MAX_RANGE_M)
    ranges = np.clip(ranges, NEAR_RANGE_M, MAX_RANGE_M).astype(np.float32)
    return ranges, valid.astype(np.uint8)


def test_range_geometry_baseline_recovers_straight_rectangular_tunnel() -> None:
    ranges, valid = _rectangular_tunnel_scan()
    result = RangeGeometryBaseline().predict(ranges, valid)
    assert abs(result["local_axis"][0]) > 0.98
    assert result["width_m"] == pytest.approx(5.0, abs=0.8)
    assert result["height_m"] == pytest.approx(4.0, abs=0.8)
    assert result["slope_deg"] == pytest.approx(0.0, abs=1.0)
    assert result["curvature_per_m"] == pytest.approx(0.0, abs=0.02)


def test_range_geometry_baseline_rejects_empty_scan() -> None:
    ranges = np.full((len(ELEVATION_DEG), AZIMUTH_COLUMNS), MAX_RANGE_M, dtype=np.float32)
    with pytest.raises(RuntimeError, match="insufficient local returns"):
        RangeGeometryBaseline().predict(ranges, np.zeros_like(ranges, dtype=np.uint8))
