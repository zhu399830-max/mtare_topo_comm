"""Pre-registered aggregate decision for the GSE validation perception gate."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.evaluation.gse_metrics import relative_geometry_improvement


def summarize_gse_perception_gate(
    *,
    learned_seed_metrics: Sequence[Mapping[str, Any]],
    calibrated_seed_metrics: Sequence[Mapping[str, Any]],
    exit_only_seed_metrics: Sequence[Mapping[str, Any]],
    nonlearning_geometry_mae: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the frozen three-seed event, geometry and association criteria."""

    if not all(len(values) == 3 for values in (learned_seed_metrics, calibrated_seed_metrics, exit_only_seed_metrics)):
        raise ValueError("the GSE perception decision requires exactly three seeds")
    learned_by_seed = {int(item["seed"]): item for item in learned_seed_metrics}
    calibrated_by_seed = {int(item["seed"]): item for item in calibrated_seed_metrics}
    baseline_by_seed = {int(item["seed"]): item for item in exit_only_seed_metrics}
    if set(learned_by_seed) != {0, 1, 2} or set(calibrated_by_seed) != {0, 1, 2} or set(baseline_by_seed) != {0, 1, 2}:
        raise ValueError("perception metrics must contain unique seeds 0/1/2")

    event_rows = []
    for seed in (0, 1, 2):
        learned_f1 = float(calibrated_by_seed[seed]["event_macro_f1"])
        baseline_f1 = float(baseline_by_seed[seed]["event"]["macro_f1"])
        if not np.isfinite(learned_f1) or not np.isfinite(baseline_f1):
            raise ValueError("event metrics must be finite")
        event_rows.append(
            {
                "seed": seed,
                "gse_calibrated_macro_f1": learned_f1,
                "exit_only_macro_f1": baseline_f1,
                "absolute_improvement": learned_f1 - baseline_f1,
            }
        )
    event_improvements = [row["absolute_improvement"] for row in event_rows]
    event_mean = float(np.mean(event_improvements))
    event_gate = {
        "per_seed": event_rows,
        "mean_absolute_improvement": event_mean,
        "required_mean_improvement": 0.05,
        "seeds_at_or_above_five_points": int(np.sum(np.asarray(event_improvements) >= 0.05)),
        "required_seeds_at_or_above_five_points": 2,
        "worst_seed_improvement": float(min(event_improvements)),
        "passed": bool(
            event_mean >= 0.05
            and int(np.sum(np.asarray(event_improvements) >= 0.05)) >= 2
            and min(event_improvements) >= 0.0
        ),
    }

    geometry_names = tuple(sorted(nonlearning_geometry_mae))
    if geometry_names != tuple(sorted(("width_m", "height_m", "slope_deg", "curvature_per_m"))):
        raise ValueError("non-learning geometry must contain the four frozen fields")
    learned_geometry = {
        name: float(np.mean([float(learned_by_seed[seed]["geometry_mae"][name]) for seed in (0, 1, 2)]))
        for name in geometry_names
    }
    geometry_gate = relative_geometry_improvement(learned_geometry, nonlearning_geometry_mae)
    geometry_gate["gse_three_seed_mean_mae"] = learned_geometry
    geometry_gate["nonlearning_mae"] = {name: float(nonlearning_geometry_mae[name]) for name in geometry_names}
    geometry_gate["per_seed_gse_mae"] = {
        str(seed): {name: float(learned_by_seed[seed]["geometry_mae"][name]) for name in geometry_names}
        for seed in (0, 1, 2)
    }

    association_rows = []
    for seed in (0, 1, 2):
        current = calibrated_by_seed[seed]
        row = {
            "seed": seed,
            "place_precision": float(current["place_association_precision"]),
            "place_false_accept_rate": float(current["place_false_accept_rate"]),
            "place_accepted": int(current["place_accepted"]),
            "exit_precision": float(current["exit_association_precision"]),
            "exit_false_accept_rate": float(current["exit_false_accept_rate"]),
            "exit_accepted": int(current["exit_accepted"]),
        }
        row["passed"] = bool(
            row["place_precision"] >= 0.98
            and row["place_false_accept_rate"] <= 0.01
            and row["place_accepted"] > 0
            and row["exit_precision"] >= 0.98
            and row["exit_false_accept_rate"] <= 0.01
            and row["exit_accepted"] > 0
        )
        association_rows.append(row)
    association_gate = {
        "minimum_precision": 0.98,
        "maximum_false_accept_rate": 0.01,
        "zero_acceptance_forbidden": True,
        "per_seed": association_rows,
        "passed": all(row["passed"] for row in association_rows),
    }
    passed = bool(event_gate["passed"] and geometry_gate["passed"] and association_gate["passed"])
    return {
        "event_gate": event_gate,
        "geometry_gate": geometry_gate,
        "association_gate": association_gate,
        "passed": passed,
        "scientific_status": "PASS_GSE_PERCEPTION_GATE_V1" if passed else "FAIL_GSE_PERCEPTION_GATE_V1",
    }


__all__ = ["summarize_gse_perception_gate"]
