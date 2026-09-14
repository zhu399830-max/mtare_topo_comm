"""Explicit current-scan B0 safety fallback for empty learned exit directions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from mtare_topo.evaluation.phase3_semantic_metrics import decode_direction_components
from mtare_topo.integration.online_topology_runtime import SemanticPrediction
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline


ELEVATION_DEG = np.arange(-15, 16, 2, dtype=np.float64)


@dataclass(frozen=True)
class EmptyDirectionFallbackAudit:
    learned_heading_count: int
    learned_empty: bool
    fallback_used: bool
    fallback_heading_count: int
    fallback_headings_robot_deg: tuple[float, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CompositeV9Audit:
    b0_branch_count: int
    b0_role_index: int
    neural_count_role_ignored: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _one_hot(size: int, index: int) -> np.ndarray:
    output = np.zeros(size, dtype=np.float64)
    output[index] = 1.0
    return output


def apply_empty_direction_fallback(
    prediction: SemanticPrediction,
    range_m: np.ndarray,
    valid_mask: np.ndarray,
    *,
    direction_threshold: float = 0.5,
    baseline: RangeExitBaseline | None = None,
) -> tuple[SemanticPrediction, EmptyDirectionFallbackAudit]:
    """Use B0 only when learned directions are empty; retain learned embedding."""

    logits = np.asarray(prediction.direction_logits, dtype=np.float64)
    learned_headings = decode_direction_components(logits, direction_threshold)
    if learned_headings:
        return prediction, EmptyDirectionFallbackAudit(
            learned_heading_count=len(learned_headings),
            learned_empty=False,
            fallback_used=False,
            fallback_heading_count=0,
            fallback_headings_robot_deg=(),
        )
    result = (baseline or RangeExitBaseline()).predict(range_m, valid_mask, ELEVATION_DEG)
    headings = tuple(float(value) for value in result["headings_robot_deg"])
    if not headings:
        return prediction, EmptyDirectionFallbackAudit(0, True, False, 0, ())
    fallback_logits = np.full(720, -20.0, dtype=np.float64)
    for heading in headings:
        fallback_logits[int(round((heading % 360.0) * 2.0)) % 720] = 20.0
    count = min(max(len(headings), 1), 6)
    role = 2 if count <= 1 else 0 if count == 2 else 1
    corrected = SemanticPrediction(
        direction_logits=fallback_logits,
        count_probabilities=_one_hot(6, count - 1),
        role_probabilities=_one_hot(3, role),
        z_role=prediction.z_role,
    )
    return corrected, EmptyDirectionFallbackAudit(
        learned_heading_count=0,
        learned_empty=True,
        fallback_used=True,
        fallback_heading_count=len(headings),
        fallback_headings_robot_deg=headings,
    )


def apply_composite_v9_semantics(
    prediction: SemanticPrediction,
    range_m: np.ndarray,
    valid_mask: np.ndarray,
    *,
    direction_threshold: float = 0.5,
    baseline: RangeExitBaseline | None = None,
) -> tuple[SemanticPrediction, EmptyDirectionFallbackAudit, CompositeV9Audit]:
    """Apply the sealed V9 contract: learned direction, B0 count, fixed role."""

    b0 = baseline or RangeExitBaseline()
    result = b0.predict(range_m, valid_mask, ELEVATION_DEG)
    headings = tuple(float(value) for value in result["headings_robot_deg"])
    count = int(result.get("branch_count", len(headings)))
    if count != len(headings) or not 0 <= count <= 8:
        raise ValueError(f"V9 B0 branch-count contract failed: count={count}, headings={len(headings)}")
    role = 2 if count <= 1 else 0 if count == 2 else 1
    logits = np.asarray(prediction.direction_logits, dtype=np.float64)
    learned = decode_direction_components(logits, direction_threshold)
    fallback_used = bool(not learned and headings)
    direction_logits = logits
    if fallback_used:
        direction_logits = np.full(720, -20.0, dtype=np.float64)
        for heading in headings:
            direction_logits[int(round((heading % 360.0) * 2.0)) % 720] = 20.0
    corrected = SemanticPrediction(
        direction_logits=direction_logits,
        count_probabilities=_one_hot(6, min(max(count, 1), 6) - 1),
        role_probabilities=_one_hot(3, role),
        z_role=prediction.z_role,
        branch_count_override=count,
    )
    fallback = EmptyDirectionFallbackAudit(
        learned_heading_count=len(learned),
        learned_empty=not learned,
        fallback_used=fallback_used,
        fallback_heading_count=len(headings) if fallback_used else 0,
        fallback_headings_robot_deg=headings if fallback_used else (),
    )
    audit = CompositeV9Audit(count, role, True)
    return corrected, fallback, audit


__all__ = [
    "CompositeV9Audit",
    "EmptyDirectionFallbackAudit",
    "apply_composite_v9_semantics",
    "apply_empty_direction_fallback",
]
