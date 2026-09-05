from __future__ import annotations

from mtare_topo.evaluation.gse_perception_gate import summarize_gse_perception_gate


def _inputs(event_improvements=(0.06, 0.07, 0.05), curvature=0.90):
    learned = [
        {
            "seed": seed,
            "geometry_mae": {
                "width_m": 0.8,
                "height_m": 0.8,
                "slope_deg": 0.8,
                "curvature_per_m": curvature,
            },
        }
        for seed in range(3)
    ]
    baseline_event = (0.40, 0.41, 0.42)
    calibrated = [
        {
            "seed": seed,
            "event_macro_f1": baseline_event[seed] + event_improvements[seed],
            "place_association_precision": 0.99,
            "place_false_accept_rate": 0.01,
            "place_accepted": 20,
            "exit_association_precision": 1.0,
            "exit_false_accept_rate": 0.0,
            "exit_accepted": 30,
        }
        for seed in range(3)
    ]
    exit_only = [
        {"seed": seed, "event": {"macro_f1": baseline_event[seed]}}
        for seed in range(3)
    ]
    geometry = {"width_m": 1.0, "height_m": 1.0, "slope_deg": 1.0, "curvature_per_m": 1.0}
    return learned, calibrated, exit_only, geometry


def test_complete_perception_gate_passes_only_all_three_subgates() -> None:
    learned, calibrated, exit_only, geometry = _inputs()
    result = summarize_gse_perception_gate(
        learned_seed_metrics=learned,
        calibrated_seed_metrics=calibrated,
        exit_only_seed_metrics=exit_only,
        nonlearning_geometry_mae=geometry,
    )
    assert result["event_gate"]["passed"] is True
    assert result["geometry_gate"]["passed"] is True
    assert result["association_gate"]["passed"] is True
    assert result["passed"] is True


def test_one_event_seed_cannot_regress() -> None:
    learned, calibrated, exit_only, geometry = _inputs(event_improvements=(0.10, 0.10, -0.01))
    result = summarize_gse_perception_gate(
        learned_seed_metrics=learned,
        calibrated_seed_metrics=calibrated,
        exit_only_seed_metrics=exit_only,
        nonlearning_geometry_mae=geometry,
    )
    assert result["event_gate"]["mean_absolute_improvement"] > 0.05
    assert result["event_gate"]["passed"] is False
    assert result["passed"] is False


def test_zero_acceptance_never_passes_association() -> None:
    learned, calibrated, exit_only, geometry = _inputs()
    calibrated[0]["place_accepted"] = 0
    result = summarize_gse_perception_gate(
        learned_seed_metrics=learned,
        calibrated_seed_metrics=calibrated,
        exit_only_seed_metrics=exit_only,
        nonlearning_geometry_mae=geometry,
    )
    assert result["association_gate"]["passed"] is False
    assert result["passed"] is False
