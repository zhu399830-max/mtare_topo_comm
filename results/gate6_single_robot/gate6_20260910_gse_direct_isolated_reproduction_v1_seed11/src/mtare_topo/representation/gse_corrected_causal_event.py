"""Contracts for learning the corrected persistent causal change-point Teacher."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


OLD_DIRECTIONAL_ENSEMBLE_MACRO_F1 = 0.6879041031973032
REQUIRED_EVENT_MACRO_F1 = OLD_DIRECTIONAL_ENSEMBLE_MACRO_F1 + 0.05
SELECTION_CHANGE_POINT_IDENTITIES = 17
REQUIRED_CHANGE_POINT_IDENTITIES_COVERED = 7


@dataclass(frozen=True)
class BinomialInterval:
    successes: int
    trials: int
    estimate: float
    lower_95: float
    upper_95: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "successes": self.successes,
            "trials": self.trials,
            "estimate": self.estimate,
            "lower_95": self.lower_95,
            "upper_95": self.upper_95,
        }


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> BinomialInterval:
    """Return a deterministic two-sided Wilson interval for identity coverage."""

    if trials <= 0 or successes < 0 or successes > trials or not math.isfinite(z) or z <= 0:
        raise ValueError("invalid binomial interval inputs")
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (probability + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(probability * (1.0 - probability) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return BinomialInterval(
        successes=successes,
        trials=trials,
        estimate=probability,
        lower_95=max(0.0, centre - radius),
        upper_95=min(1.0, centre + radius),
    )


def corrected_causal_event_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Pre-registered selection gate for the corrected causal event head."""

    selection = metrics.get("structural_selection") or {}
    coverage = metrics.get("identity_coverage") or {}
    transition = coverage.get("geometry_transition") or {}
    covered_transition = int(transition.get("covered_identities", 0))
    teacher_transition = int(transition.get("teacher_identities", 0))
    requirements = {
        "event_macro_f1_vs_old_directional_plus_0p05": float(
            metrics.get("event", {}).get("macro_f1", 0.0)
        )
        >= REQUIRED_EVENT_MACRO_F1,
        "structural_precision": float(selection.get("precision", 0.0)) >= 0.98,
        "structural_false_accept": float(selection.get("false_accept_rate", 1.0)) <= 0.01,
        "structural_recall": float(selection.get("recall", 0.0)) >= 0.40,
        "junction_identity_coverage": float(
            coverage.get("junction", {}).get("correct_class_identity_coverage", 0.0)
        )
        >= 0.90,
        "terminal_identity_coverage": float(
            coverage.get("terminal", {}).get("correct_class_identity_coverage", 0.0)
        )
        >= 0.90,
        "turn_identity_coverage": float(
            coverage.get("turn", {}).get("correct_class_identity_coverage", 0.0)
        )
        >= 0.40,
        "change_point_selection_identity_count_exact_17": teacher_transition
        == SELECTION_CHANGE_POINT_IDENTITIES,
        "change_point_identity_coverage_at_least_7_of_17": covered_transition
        >= REQUIRED_CHANGE_POINT_IDENTITIES_COVERED,
    }
    interval = (
        wilson_interval(covered_transition, teacher_transition).to_dict()
        if teacher_transition > 0
        else None
    )
    return {
        "passed": all(requirements.values()),
        "requirements": requirements,
        "change_point_identity_coverage_wilson_95": interval,
    }


__all__ = [
    "BinomialInterval",
    "OLD_DIRECTIONAL_ENSEMBLE_MACRO_F1",
    "REQUIRED_CHANGE_POINT_IDENTITIES_COVERED",
    "REQUIRED_EVENT_MACRO_F1",
    "SELECTION_CHANGE_POINT_IDENTITIES",
    "corrected_causal_event_gate",
    "wilson_interval",
]
