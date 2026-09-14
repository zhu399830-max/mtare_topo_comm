"""World-robust calibration for the physics-guided slope residual.

The residual network is deliberately kept separate from this calibration.  A
single scale in ``[0, 1]`` is selected on the frozen C07--C08 development
worlds by maximizing the worst parent-level improvement over the analytic
five-frame prior.  This makes the deployment rule capable of retaining part
of the analytic expert when a full learned residual is too aggressive.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SlopeResidualRiskCalibration:
    residual_scale: float
    grid_index: int
    grid_denominator: int
    worst_parent_relative_improvement: float
    mean_parent_relative_improvement: float
    parent_relative_improvement: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def apply_residual_scale(
    prior_slope_deg: np.ndarray,
    corrected_slope_deg: np.ndarray,
    residual_scale: float,
) -> np.ndarray:
    """Under-relax a learned correction while preserving the analytic prior."""

    prior = np.asarray(prior_slope_deg, dtype=np.float64)
    corrected = np.asarray(corrected_slope_deg, dtype=np.float64)
    scale = float(residual_scale)
    if prior.shape != corrected.shape or prior.ndim != 1:
        raise ValueError("prior and corrected slope must be equal one-dimensional arrays")
    if not np.all(np.isfinite(prior)) or not np.all(np.isfinite(corrected)):
        raise ValueError("slope arrays must be finite")
    if not np.isfinite(scale) or scale < 0.0 or scale > 1.0:
        raise ValueError("residual scale must be finite and within [0, 1]")
    return (prior + scale * (corrected - prior)).astype(np.float32)


def select_maximin_residual_scale(
    seed_outputs: Sequence[Mapping[str, np.ndarray]],
    *,
    grid_denominator: int = 100,
) -> SlopeResidualRiskCalibration:
    """Select one shared scale using parent-level, seed-averaged maximin gain.

    Ties are resolved by larger mean parent improvement and then the smaller
    grid index.  The input must contain the same non-empty parent set for every
    seed; seeds are repeated measurements rather than independent worlds.
    """

    if not seed_outputs or grid_denominator <= 0:
        raise ValueError("non-empty seed outputs and a positive grid denominator are required")
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    expected_parents: set[str] | None = None
    for row in seed_outputs:
        target = np.asarray(row["target_slope_deg"], dtype=np.float64)
        prior = np.asarray(row["five_frame_prior_slope_deg"], dtype=np.float64)
        corrected = np.asarray(row["corrected_slope_deg"], dtype=np.float64)
        parents = np.asarray(row["parent_id"]).astype(str)
        if (
            target.ndim != 1
            or target.shape != prior.shape
            or target.shape != corrected.shape
            or target.shape != parents.shape
            or len(target) == 0
            or not np.all(np.isfinite(target))
            or not np.all(np.isfinite(prior))
            or not np.all(np.isfinite(corrected))
        ):
            raise ValueError("each seed must provide aligned finite slope and parent arrays")
        parent_set = set(parents.tolist())
        if not parent_set or (expected_parents is not None and parent_set != expected_parents):
            raise ValueError("all seeds must contain the same non-empty parent set")
        expected_parents = parent_set
        prepared.append((target, prior, corrected - prior, parents))

    assert expected_parents is not None
    best_key: tuple[float, float, int] | None = None
    best_index = -1
    best_parent_values: dict[str, float] = {}
    for index in range(grid_denominator + 1):
        scale = index / grid_denominator
        parent_values: dict[str, float] = {}
        for parent in sorted(expected_parents):
            repeated = []
            for target, prior, residual, parents in prepared:
                mask = parents == parent
                baseline_mae = float(np.mean(np.abs(prior[mask] - target[mask])))
                if baseline_mae <= 0.0:
                    raise ValueError(f"parent has zero analytic baseline error: {parent}")
                calibrated_mae = float(np.mean(np.abs(prior[mask] + scale * residual[mask] - target[mask])))
                repeated.append((baseline_mae - calibrated_mae) / baseline_mae)
            parent_values[parent] = float(np.mean(repeated))
        worst = min(parent_values.values())
        mean = float(np.mean(list(parent_values.values())))
        key = (worst, mean, -index)
        if best_key is None or key > best_key:
            best_key = key
            best_index = index
            best_parent_values = parent_values

    assert best_key is not None
    return SlopeResidualRiskCalibration(
        residual_scale=best_index / grid_denominator,
        grid_index=best_index,
        grid_denominator=grid_denominator,
        worst_parent_relative_improvement=float(best_key[0]),
        mean_parent_relative_improvement=float(best_key[1]),
        parent_relative_improvement=best_parent_values,
    )


__all__ = [
    "SlopeResidualRiskCalibration",
    "apply_residual_scale",
    "select_maximin_residual_scale",
]
