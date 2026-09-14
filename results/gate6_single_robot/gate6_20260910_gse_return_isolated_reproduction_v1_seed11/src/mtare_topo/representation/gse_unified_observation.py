"""Fail-closed integration of the final slope estimate into GSE observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


OBSERVATION_FEATURE_DIM = 146
SLOPE_FEATURE_INDEX = 10
SLOPE_NORMALIZATION_DEG = 45.0


@dataclass(frozen=True)
class UnifiedSlopeAudit:
    observations: int
    changed_columns: tuple[int, ...]
    unchanged_columns_byte_exact: bool
    maximum_absolute_slope_deg: float
    mean_absolute_change_deg: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def align_by_global_sequence_index(
    observation_global_index: np.ndarray,
    source_global_index: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    """Align one scalar value to every observation without positional assumptions."""

    observation_index = np.asarray(observation_global_index, dtype=np.int64)
    source_index = np.asarray(source_global_index, dtype=np.int64)
    values = np.asarray(source_values, dtype=np.float32)
    if (
        observation_index.ndim != 1
        or source_index.shape != observation_index.shape
        or values.shape != observation_index.shape
        or len(np.unique(observation_index)) != len(observation_index)
        or len(np.unique(source_index)) != len(source_index)
        or not np.array_equal(np.sort(observation_index), np.sort(source_index))
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("unified slope global sequence identity contract drift")
    order = np.argsort(source_index, kind="stable")
    positions = np.searchsorted(source_index[order], observation_index)
    if np.any(positions >= len(order)) or not np.array_equal(source_index[order][positions], observation_index):
        raise RuntimeError("unified slope alignment failed")
    return values[order][positions]


def replace_normalized_slope(
    observation_features: np.ndarray,
    observation_global_index: np.ndarray,
    slope_global_index: np.ndarray,
    corrected_slope_deg: np.ndarray,
) -> tuple[np.ndarray, UnifiedSlopeAudit]:
    """Replace only the normalized slope column and prove all others unchanged."""

    original = np.asarray(observation_features)
    if (
        original.ndim != 2
        or original.shape[1] != OBSERVATION_FEATURE_DIM
        or original.dtype != np.float32
        or not np.all(np.isfinite(original))
    ):
        raise ValueError("frozen GSE observation feature contract drift")
    aligned = align_by_global_sequence_index(
        observation_global_index, slope_global_index, corrected_slope_deg
    )
    if np.any(np.abs(aligned) > SLOPE_NORMALIZATION_DEG):
        raise ValueError("corrected slope exceeds the frozen normalization domain")
    unified = original.copy()
    unified[:, SLOPE_FEATURE_INDEX] = aligned / SLOPE_NORMALIZATION_DEG
    changed = tuple(int(value) for value in np.flatnonzero(np.any(original != unified, axis=0)))
    unchanged = (
        original[:, :SLOPE_FEATURE_INDEX].tobytes()
        == unified[:, :SLOPE_FEATURE_INDEX].tobytes()
        and original[:, SLOPE_FEATURE_INDEX + 1 :].tobytes()
        == unified[:, SLOPE_FEATURE_INDEX + 1 :].tobytes()
    )
    if any(column != SLOPE_FEATURE_INDEX for column in changed) or not unchanged:
        raise RuntimeError("unified observation changed a non-slope feature")
    audit = UnifiedSlopeAudit(
        observations=len(original),
        changed_columns=changed,
        unchanged_columns_byte_exact=unchanged,
        maximum_absolute_slope_deg=float(np.max(np.abs(aligned))),
        mean_absolute_change_deg=float(
            np.mean(np.abs(original[:, SLOPE_FEATURE_INDEX] * SLOPE_NORMALIZATION_DEG - aligned))
        ),
    )
    return unified, audit


__all__ = [
    "OBSERVATION_FEATURE_DIM",
    "SLOPE_FEATURE_INDEX",
    "SLOPE_NORMALIZATION_DEG",
    "UnifiedSlopeAudit",
    "align_by_global_sequence_index",
    "replace_normalized_slope",
]
