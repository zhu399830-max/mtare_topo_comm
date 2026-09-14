"""Dense circular peak-field representation for executable exit geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BEARING_BINS = 180
BIN_WIDTH_DEG = 2.0


def _signed_circular_delta_deg(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=np.float64) - np.asarray(reference, dtype=np.float64) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class CircularExitGeometryField:
    presence: np.ndarray
    heading_residual_deg: np.ndarray
    opening_width_m: np.ndarray
    width_valid_mask: np.ndarray
    vertical_profile_m: np.ndarray


def heading_unit_to_bearing_bins(heading_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical float64 conversion shared by encoding and audit code."""

    heading = np.asarray(heading_unit, dtype=np.float64)
    if heading.ndim < 1 or heading.shape[-1] != 2 or not np.all(np.isfinite(heading)):
        raise ValueError("heading units must be finite with final dimension two")
    angle = np.degrees(np.arctan2(heading[..., 0], heading[..., 1])) % 360.0
    bins = np.floor((angle + BIN_WIDTH_DEG / 2.0) / BIN_WIDTH_DEG).astype(np.int64) % BEARING_BINS
    centers = bins.astype(np.float64) * BIN_WIDTH_DEG
    residual = _signed_circular_delta_deg(angle, centers)
    return angle, bins, residual


def encode_exit_geometry_peaks(
    *,
    exit_mask: np.ndarray,
    heading_unit: np.ndarray,
    opening_width_m: np.ndarray,
    width_valid_mask: np.ndarray,
    vertical_profile_m: np.ndarray,
) -> CircularExitGeometryField:
    """Rasterize each visible exit at its unique nearest 2-degree bearing bin."""

    mask = np.asarray(exit_mask, dtype=np.bool_)
    heading = np.asarray(heading_unit, dtype=np.float64)
    width = np.asarray(opening_width_m, dtype=np.float32)
    width_mask = np.asarray(width_valid_mask, dtype=np.bool_)
    profile = np.asarray(vertical_profile_m, dtype=np.float32)
    if (
        mask.ndim != 2
        or heading.shape != mask.shape + (2,)
        or width.shape != mask.shape
        or width_mask.shape != mask.shape
        or profile.shape != mask.shape + (4,)
        or np.any(width_mask & ~mask)
        or not np.all(np.isfinite(heading))
        or not np.all(np.isfinite(width))
        or not np.all(np.isfinite(profile))
    ):
        raise ValueError("exit peak-field input contract drift")
    norm = np.linalg.norm(heading[mask], axis=1)
    if len(norm) == 0 or not np.allclose(norm, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("visible exit headings must be unit vectors")
    angle, bins, residual = heading_unit_to_bearing_bins(heading)
    if np.any(np.abs(residual[mask]) > BIN_WIDTH_DEG / 2.0 + 1e-9):
        raise RuntimeError("heading residual escaped its bearing bin")

    rows = len(mask)
    presence = np.zeros((rows, BEARING_BINS), dtype=np.bool_)
    residual_field = np.zeros((rows, BEARING_BINS), dtype=np.float32)
    width_field = np.zeros((rows, BEARING_BINS), dtype=np.float32)
    width_valid_field = np.zeros((rows, BEARING_BINS), dtype=np.bool_)
    profile_field = np.zeros((rows, BEARING_BINS, 4), dtype=np.float32)
    for row in range(rows):
        slots = np.flatnonzero(mask[row])
        active_bins = bins[row, slots]
        if len(np.unique(active_bins)) != len(active_bins):
            raise RuntimeError(f"multiple exits collide in one bearing bin at row {row}")
        presence[row, active_bins] = True
        residual_field[row, active_bins] = residual[row, slots].astype(np.float32)
        width_field[row, active_bins] = width[row, slots]
        width_valid_field[row, active_bins] = width_mask[row, slots]
        profile_field[row, active_bins] = profile[row, slots]
    return CircularExitGeometryField(
        presence=presence,
        heading_residual_deg=residual_field,
        opening_width_m=width_field,
        width_valid_mask=width_valid_field,
        vertical_profile_m=profile_field,
    )


def decode_exit_geometry_peaks(field: CircularExitGeometryField) -> list[dict[str, np.ndarray]]:
    """Decode deterministic circular local maxima represented by binary Teacher peaks."""

    presence = np.asarray(field.presence, dtype=np.bool_)
    residual = np.asarray(field.heading_residual_deg, dtype=np.float64)
    width = np.asarray(field.opening_width_m, dtype=np.float32)
    width_mask = np.asarray(field.width_valid_mask, dtype=np.bool_)
    profile = np.asarray(field.vertical_profile_m, dtype=np.float32)
    if (
        presence.ndim != 2
        or presence.shape[1] != BEARING_BINS
        or residual.shape != presence.shape
        or width.shape != presence.shape
        or width_mask.shape != presence.shape
        or profile.shape != presence.shape + (4,)
        or np.any(width_mask & ~presence)
    ):
        raise ValueError("exit peak field is invalid")
    decoded: list[dict[str, np.ndarray]] = []
    for row in range(len(presence)):
        bins = np.flatnonzero(presence[row])
        angle = (bins.astype(np.float64) * BIN_WIDTH_DEG + residual[row, bins]) % 360.0
        decoded.append({
            "bearing_bin": bins,
            "heading_deg": angle,
            "opening_width_m": width[row, bins],
            "width_valid_mask": width_mask[row, bins],
            "vertical_profile_m": profile[row, bins],
        })
    return decoded


def circular_component_count(mask: np.ndarray) -> np.ndarray:
    """Count connected true components on a circular one-dimensional field."""

    value = np.asarray(mask, dtype=np.bool_)
    if value.ndim != 2 or value.shape[1] != BEARING_BINS:
        raise ValueError("circular component mask must be [N,180]")
    starts = value & ~np.roll(value, 1, axis=1)
    count = starts.sum(axis=1).astype(np.int64)
    count[np.all(value, axis=1)] = 1
    return count


def circular_roll_field(field: CircularExitGeometryField, shift_bins: int) -> CircularExitGeometryField:
    """Apply an exact robot-yaw rotation by an integer number of 2-degree bins."""

    shift = int(shift_bins)
    return CircularExitGeometryField(
        presence=np.roll(field.presence, shift, axis=1),
        heading_residual_deg=np.roll(field.heading_residual_deg, shift, axis=1),
        opening_width_m=np.roll(field.opening_width_m, shift, axis=1),
        width_valid_mask=np.roll(field.width_valid_mask, shift, axis=1),
        vertical_profile_m=np.roll(field.vertical_profile_m, shift, axis=1),
    )


__all__ = [
    "BEARING_BINS",
    "BIN_WIDTH_DEG",
    "CircularExitGeometryField",
    "circular_component_count",
    "circular_roll_field",
    "decode_exit_geometry_peaks",
    "encode_exit_geometry_peaks",
    "heading_unit_to_bearing_bins",
]
