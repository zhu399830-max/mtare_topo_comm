"""Combine the immutable original C09 gate with corrected slope evidence."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mtare_topo.evaluation.gse_perception_gate import summarize_gse_perception_gate


def summarize_corrected_perception_gate(
    original_gate: Mapping[str, Any],
    corrective: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        original_gate.get("overall_status") != "FAIL_GSE_PERCEPTION_GATE_V1"
        or original_gate.get("event_gate", {}).get("passed") is not True
        or original_gate.get("association_gate", {}).get("passed") is not True
        or original_gate.get("geometry_gate", {}).get("passed") is not False
    ):
        raise ValueError("corrective gate requires the sealed slope-only original C09 failure")
    original_geometry = original_gate["geometry_gate"]
    # ``summarize_gse_perception_gate`` seals the per-field values under
    # ``per_field_relative_improvement``.  The first corrective runner used
    # the pre-summary draft key here and therefore stopped before it could
    # report the already-computed scientific per-world failure.
    relative = original_geometry.get("per_field_relative_improvement", {})
    if not (
        float(relative.get("slope_deg", 0.0)) < -0.05
        and all(float(relative[name]) >= -0.05 for name in ("width_m", "height_m", "curvature_per_m"))
    ):
        raise ValueError("original C09 failure is not isolated to slope")
    seed_rows = corrective.get("seeds")
    if not isinstance(seed_rows, list) or {int(row["seed"]) for row in seed_rows} != {0, 1, 2}:
        raise ValueError("corrective C09 evidence requires unique seeds 0/1/2")
    corrected_by_seed = {int(row["seed"]): row for row in seed_rows}
    learned = []
    for seed in (0, 1, 2):
        geometry = dict(original_geometry["per_seed_gse_mae"][str(seed)])
        geometry["slope_deg"] = float(corrected_by_seed[seed]["corrected_mae_deg"])
        learned.append({"seed": seed, "geometry_mae": geometry, "axis": original_gate["learned_axis_diagnostic"][str(seed)]})
    event_by_seed = {int(row["seed"]): row for row in original_gate["event_gate"]["per_seed"]}
    association_by_seed = {int(row["seed"]): row for row in original_gate["association_gate"]["per_seed"]}
    calibrated = []
    exit_only = []
    for seed in (0, 1, 2):
        event = event_by_seed[seed]
        association = association_by_seed[seed]
        calibrated.append(
            {
                "seed": seed,
                "event_macro_f1": event["gse_calibrated_macro_f1"],
                "place_association_precision": association["place_precision"],
                "place_false_accept_rate": association["place_false_accept_rate"],
                "place_accepted": association["place_accepted"],
                "exit_association_precision": association["exit_precision"],
                "exit_false_accept_rate": association["exit_false_accept_rate"],
                "exit_accepted": association["exit_accepted"],
            }
        )
        exit_only.append({"seed": seed, "event": {"macro_f1": event["exit_only_macro_f1"]}})
    main = summarize_gse_perception_gate(
        learned_seed_metrics=learned,
        calibrated_seed_metrics=calibrated,
        exit_only_seed_metrics=exit_only,
        nonlearning_geometry_mae=original_geometry["nonlearning_mae"],
    )
    improvements = np.asarray(
        [float(corrected_by_seed[seed]["relative_improvement_over_five_frame_prior"]) for seed in (0, 1, 2)],
        dtype=np.float64,
    )
    worlds = sorted(
        set.intersection(
            *[
                {str(row["parent_id"]) for row in corrected_by_seed[seed]["per_world"]}
                for seed in (0, 1, 2)
            ]
        )
    )
    if len(worlds) != 10:
        raise ValueError("corrective C09 evidence must contain ten common worlds")
    world_rows = []
    for parent in worlds:
        rows = [
            next(row for row in corrected_by_seed[seed]["per_world"] if row["parent_id"] == parent)
            for seed in (0, 1, 2)
        ]
        prior = float(np.mean([float(row["five_frame_prior_mae_deg"]) for row in rows]))
        corrected = float(np.mean([float(row["corrected_mae_deg"]) for row in rows]))
        world_rows.append(
            {
                "parent_id": parent,
                "five_frame_prior_mae_deg": prior,
                "corrected_three_seed_mean_mae_deg": corrected,
                "relative_improvement": (prior - corrected) / prior,
            }
        )
    learning_gate = {
        "per_seed_relative_improvement": improvements.tolist(),
        "mean_relative_improvement": float(improvements.mean()),
        "required_mean_relative_improvement": 0.05,
        "seeds_at_or_above_five_percent": int(np.sum(improvements >= 0.05)),
        "required_seeds_at_or_above_five_percent": 2,
        "all_seed_nonregression": bool(np.all(improvements >= 0.0)),
        "per_world_three_seed_mean": world_rows,
        "maximum_allowed_world_regression": 0.05,
        "all_world_regression_within_five_percent": all(row["relative_improvement"] >= -0.05 for row in world_rows),
    }
    learning_gate["passed"] = bool(
        learning_gate["mean_relative_improvement"] >= 0.05
        and learning_gate["seeds_at_or_above_five_percent"] >= 2
        and learning_gate["all_seed_nonregression"]
        and learning_gate["all_world_regression_within_five_percent"]
    )
    passed = bool(main["passed"] and learning_gate["passed"])
    return {
        "schema_version": "gse_corrected_perception_gate_v1",
        "overall_status": "PASS_GSE_CORRECTED_PERCEPTION_GATE_V1" if passed else "FAIL_GSE_CORRECTED_PERCEPTION_GATE_V1",
        "passed": passed,
        "event_gate": main["event_gate"],
        "geometry_gate": main["geometry_gate"],
        "association_gate": main["association_gate"],
        "slope_learning_gate": learning_gate,
        "learned_axis_diagnostic": main.get("learned_axis_diagnostic", original_gate["learned_axis_diagnostic"]),
    }


__all__ = ["summarize_corrected_perception_gate"]
