from __future__ import annotations

import math
from typing import Any

import numpy as np


def _circular_runs(active: np.ndarray) -> list[np.ndarray]:
    active = np.asarray(active, dtype=bool)
    if not active.any():
        return []
    if active.all():
        return [np.arange(len(active), dtype=np.int64)]
    start = int(np.where(~active)[0][0])
    rolled = np.roll(active, -start)
    runs: list[np.ndarray] = []
    i = 0
    while i < len(rolled):
        if rolled[i]:
            j = i
            while j < len(rolled) and rolled[j]:
                j += 1
            runs.append((np.arange(i, j, dtype=np.int64) + start) % len(active))
            i = j
        else:
            i += 1
    return runs


def extract_exit_sectors(
    traversable: np.ndarray,
    distance: np.ndarray,
    direction_count: int = 32,
    max_sectors: int = 8,
    min_distance: float = 0.05,
) -> dict[str, Any]:
    """Convert directional reach facts to circular contiguous exit sectors.

    No bins are added. A sector is exactly the circular union of adjacent
    traversable bins that have positive reachable distance. The soft value is
    the angular overlap between each direction-bin interval and that sector;
    this preserves one-bin narrow sectors without inventing width.
    """
    active = (np.asarray(traversable) >= 0.5) & (np.asarray(distance) > min_distance)
    runs = _circular_runs(active)
    runs = sorted(runs, key=lambda row: (-len(row), int(row[0])))[:max_sectors]
    soft = np.zeros(direction_count, dtype=np.float32)
    binary = np.zeros(direction_count, dtype=np.float32)
    centers = np.zeros(max_sectors, dtype=np.float32)
    widths = np.zeros(max_sectors, dtype=np.float32)
    lengths = np.zeros(max_sectors, dtype=np.float32)
    valid = np.zeros(max_sectors, dtype=np.float32)
    bin_width = 2.0 * math.pi / direction_count
    for slot, run in enumerate(runs):
        if not len(run):
            continue
        # Bins are represented by their full angular intervals. Circular
        # ordering is unwrapped around the first bin for stable means.
        unwrapped = np.unwrap((run * bin_width + 0.5 * bin_width) % (2 * math.pi))
        center = float(np.mod(np.mean(unwrapped), 2 * math.pi))
        width = float(len(run) * bin_width)
        centers[slot] = center
        widths[slot] = width
        lengths[slot] = float(np.max(np.asarray(distance)[run]))
        valid[slot] = 1.0
        for idx in run:
            soft[int(idx)] = 1.0
            binary[int(idx)] = 1.0
    return {
        "soft": soft,
        "binary": binary,
        "centers": centers,
        "widths": widths,
        "lengths": lengths,
        "valid_mask": valid,
        "count": int(valid.sum()),
        "source_active_bins": active.astype(np.uint8),
    }


def canonical_role_descriptor(
    exit_soft: np.ndarray,
    distance: np.ndarray,
    sector_widths: np.ndarray,
    sector_lengths: np.ndarray,
    sector_valid: np.ndarray,
    dimension: int = 64,
) -> np.ndarray:
    """Create a fixed rotation-canonical descriptor without world heading.

    Circular autocorrelations remove the arbitrary robot-frame phase. Sorted
    sector widths/lengths preserve local connection multiplicity and scale.
    The descriptor is deliberately a teacher representation, not an online
    input and not a replacement for the directional field.
    """
    def circular_autocorrelation(values: np.ndarray, keep: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        corr = np.real(np.fft.ifft(np.abs(np.fft.fft(values)) ** 2))
        corr = np.maximum(corr, 0.0)
        corr = corr / max(float(np.linalg.norm(corr)), 1e-8)
        return corr[:keep].astype(np.float32)

    widths = np.sort(np.asarray(sector_widths)[np.asarray(sector_valid) > 0])[::-1]
    lengths = np.sort(np.asarray(sector_lengths)[np.asarray(sector_valid) > 0])[::-1]
    widths = np.pad(widths[:8], (0, max(0, 8 - len(widths))))[:8] / (2.0 * math.pi)
    lengths = np.pad(lengths[:8], (0, max(0, 8 - len(lengths))))[:8]
    descriptor = np.concatenate([
        circular_autocorrelation(exit_soft, 32),
        circular_autocorrelation(distance, 16),
        widths.astype(np.float32),
        lengths.astype(np.float32),
    ]).astype(np.float32)
    if len(descriptor) != dimension:
        raise ValueError(f"canonical role dimension mismatch: {len(descriptor)} != {dimension}")
    return descriptor / max(float(np.linalg.norm(descriptor)), 1e-8)


def circular_shift(values: np.ndarray, yaw_delta_rad: float) -> np.ndarray:
    values = np.asarray(values)
    shift = int(np.rint(float(yaw_delta_rad) / (2.0 * math.pi) * len(values)))
    return np.roll(values, shift)
