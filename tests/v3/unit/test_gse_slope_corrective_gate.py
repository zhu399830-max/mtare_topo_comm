from __future__ import annotations

from copy import deepcopy

from mtare_topo.evaluation.gse_slope_corrective_gate import summarize_corrected_perception_gate


def _original():
    per_seed = {str(seed): {"width_m": 1.0, "height_m": 1.0, "slope_deg": 3.0, "curvature_per_m": 1.0} for seed in range(3)}
    return {
        "overall_status": "FAIL_GSE_PERCEPTION_GATE_V1",
        "event_gate": {"passed": True, "per_seed": [{"seed": seed, "gse_calibrated_macro_f1": 0.7, "exit_only_macro_f1": 0.6} for seed in range(3)]},
        "association_gate": {"passed": True, "per_seed": [{"seed": seed, "place_precision": 0.99, "place_false_accept_rate": 0.005, "place_accepted": 10, "exit_precision": 0.99, "exit_false_accept_rate": 0.005, "exit_accepted": 10} for seed in range(3)]},
        "geometry_gate": {"passed": False, "per_field_relative_improvement": {"width_m": 0.2, "height_m": 0.2, "slope_deg": -1.0, "curvature_per_m": 0.2}, "per_seed_gse_mae": per_seed, "nonlearning_mae": {"width_m": 2.0, "height_m": 2.0, "slope_deg": 2.0, "curvature_per_m": 2.0}},
        "learned_axis_diagnostic": {str(seed): {} for seed in range(3)},
    }


def _corrective(improvement=0.5):
    worlds = [{"parent_id": f"S{index:02d}_x_C09", "five_frame_prior_mae_deg": 2.0, "corrected_mae_deg": 1.0} for index in range(1, 11)]
    return {"seeds": [{"seed": seed, "corrected_mae_deg": 1.0, "relative_improvement_over_five_frame_prior": improvement, "per_world": deepcopy(worlds)} for seed in range(3)]}


def test_corrected_gate_passes_only_when_main_and_learning_gates_pass():
    result = summarize_corrected_perception_gate(_original(), _corrective())
    assert result["passed"] is True
    assert result["geometry_gate"]["passed"] is True
    assert result["slope_learning_gate"]["passed"] is True


def test_corrected_gate_rejects_no_learned_gain_even_if_main_geometry_passes():
    result = summarize_corrected_perception_gate(_original(), _corrective(improvement=0.0))
    assert result["geometry_gate"]["passed"] is True
    assert result["slope_learning_gate"]["passed"] is False
    assert result["passed"] is False


def test_corrected_gate_rejects_legacy_draft_relative_improvement_key():
    original = _original()
    original["geometry_gate"]["relative_improvement"] = original["geometry_gate"].pop(
        "per_field_relative_improvement"
    )
    try:
        summarize_corrected_perception_gate(original, _corrective())
    except ValueError as error:
        assert str(error) == "original C09 failure is not isolated to slope"
    else:
        raise AssertionError("the sealed perception-gate schema must be required")
