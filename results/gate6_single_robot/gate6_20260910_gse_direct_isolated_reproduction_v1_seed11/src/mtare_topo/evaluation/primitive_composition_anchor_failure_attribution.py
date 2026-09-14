"""Pure diagnostics for composition-anchor C07 failure attribution."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


DISTANCE_EDGES_M = np.asarray((0.0, 5.0, 10.0, 20.0, 35.0, 50.0, np.inf))
CORRECTION_EDGES_M = np.asarray((0.0, 1.0, 2.0, 5.0, 10.0, 20.0, np.inf))
SLOPE_EDGES_DEG = np.asarray((0.0, 5.0, 10.0, 20.0, 90.0001))
SEPARATION_EDGES_M = np.asarray((0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf))


def symmetric_negative_pair_distance(points: torch.Tensor) -> torch.Tensor:
    """Return an exactly symmetric score whose rank is nearest-first."""

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("pair-distance points must be [B,N,3]")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("pair-distance points must be finite")
    delta = points[:, :, None] - points[:, None, :]
    score = -torch.linalg.vector_norm(delta, dim=-1)
    score = 0.5 * (score + score.transpose(1, 2))
    if not torch.equal(score, score.transpose(1, 2)):
        raise RuntimeError("pair-distance score is not exactly symmetric")
    return score


def binned_error_summary(
    raw_error_m: np.ndarray,
    anchor_error_m: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray,
) -> list[dict]:
    raw = np.asarray(raw_error_m, dtype=np.float64)
    anchor = np.asarray(anchor_error_m, dtype=np.float64)
    value = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if raw.shape != anchor.shape or raw.shape != value.shape or raw.ndim != 1:
        raise ValueError("error-stratum arrays must be aligned vectors")
    if len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("error-stratum edges must be strictly increasing")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(anchor)):
        raise ValueError("error-stratum errors must be finite")
    result = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (value >= lower) & (value < upper)
        count = int(np.count_nonzero(mask))
        raw_mean = float(np.mean(raw[mask])) if count else None
        anchor_mean = float(np.mean(anchor[mask])) if count else None
        improvement = (
            (raw_mean - anchor_mean) / raw_mean
            if count and raw_mean is not None and raw_mean > 0.0 else None
        )
        result.append({
            "lower_inclusive": float(lower),
            "upper_exclusive": None if np.isinf(upper) else float(upper),
            "count": count,
            "raw_mean_m": raw_mean,
            "anchor_mean_m": anchor_mean,
            "relative_improvement": improvement,
        })
    return result


def categorical_error_summary(
    raw_error_m: np.ndarray,
    anchor_error_m: np.ndarray,
    categories: np.ndarray,
    labels: Mapping[int, str],
) -> list[dict]:
    raw = np.asarray(raw_error_m, dtype=np.float64)
    anchor = np.asarray(anchor_error_m, dtype=np.float64)
    category = np.asarray(categories)
    if raw.shape != anchor.shape or raw.shape != category.shape or raw.ndim != 1:
        raise ValueError("categorical error arrays must be aligned vectors")
    result = []
    for code, label in sorted(labels.items()):
        mask = category == code
        count = int(np.count_nonzero(mask))
        raw_mean = float(np.mean(raw[mask])) if count else None
        anchor_mean = float(np.mean(anchor[mask])) if count else None
        result.append({
            "code": int(code), "label": label, "count": count,
            "raw_mean_m": raw_mean, "anchor_mean_m": anchor_mean,
            "relative_improvement": (
                (raw_mean - anchor_mean) / raw_mean
                if count and raw_mean is not None and raw_mean > 0 else None
            ),
        })
    return result


def diagnose_attribution(seed_results: list[dict]) -> dict:
    if len(seed_results) != 3:
        raise ValueError("composition-anchor attribution requires three seeds")

    def passes(condition: str) -> int:
        return sum(
            int(row["conditions"][condition]["safe"]["true_positive"] > 0)
            for row in seed_results
        )

    teacher_distance = passes("proposal_oracle_teacher_anchor_distance")
    teacher_gaussian = passes("proposal_oracle_teacher_anchor_gaussian")
    teacher_safe = passes("proposal_oracle_teacher_anchor_safe")
    predicted_distance = passes("proposal_oracle_predicted_anchor_distance")
    predicted_compatibility = passes("proposal_oracle_model_compatibility")
    deployed = passes("deployed_model_safe")
    if teacher_distance < 2:
        diagnosis = "CONSTRUCTION_ANCHOR_TARGET_DOES_NOT_SEPARATE_C07_CONNECTIONS"
        decision = "STOP_COMPOSITION_ANCHOR_TARGET_AND_REASSESS_TEACHER"
    elif predicted_distance < 2:
        diagnosis = "PREDICTED_COMPOSITION_ANCHOR_COORDINATES_ARE_PRIMARY_BOTTLENECK"
        decision = "STOP_GLOBAL_ENDPOINT_ANCHOR_REGRESSION_AND_DESIGN_OBSERVABLE_LOCAL_COMPOSITION_INTERFACE"
    elif teacher_gaussian < 2 or predicted_compatibility < 2:
        diagnosis = "COMPOSITION_ANCHOR_SCALE_OR_COMPATIBILITY_CALIBRATION_IS_PRIMARY_BOTTLENECK"
        decision = "ALLOW_SCALE_CALIBRATION_ONLY_READINESS"
    elif teacher_safe < 2 or deployed < 2:
        diagnosis = "ENDPOINT_EVIDENCE_CALIBRATION_IS_PRIMARY_BOTTLENECK"
        decision = "ALLOW_ENDPOINT_EVIDENCE_CALIBRATION_ONLY_READINESS"
    else:
        diagnosis = "COMPOSITION_ANCHOR_FAILURE_NOT_EXPLAINED_BY_PRE_REGISTERED_FACTORS"
        decision = "STOP_AND_REASSESS_PRIMITIVE_RELATION_RESEARCH_INTERFACE"
    return {
        "diagnosis": diagnosis, "decision": decision,
        "safe_passing_seeds": {
            "teacher_anchor_distance": teacher_distance,
            "teacher_anchor_gaussian": teacher_gaussian,
            "teacher_anchor_safe": teacher_safe,
            "predicted_anchor_distance": predicted_distance,
            "predicted_anchor_compatibility": predicted_compatibility,
            "deployed_model_safe": deployed,
        },
    }


__all__ = [
    "CORRECTION_EDGES_M", "DISTANCE_EDGES_M", "SEPARATION_EDGES_M",
    "SLOPE_EDGES_DEG", "binned_error_summary", "categorical_error_summary",
    "diagnose_attribution", "symmetric_negative_pair_distance",
]
