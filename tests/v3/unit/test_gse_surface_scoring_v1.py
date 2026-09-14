"""Independent synthetic scoring contracts, not trained model performance."""
from dataclasses import replace, fields
import json

import numpy as np
import pytest
import torch

from mtare_topo.representation.gse_surface_scoring_v1 import (
    SurfaceScoringConfigV1, score_surface_predictions,
)
from tests.v3.unit.test_gse_surface_losses_v1 import prediction, targets


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(**changes):
    # All values explicit synthetic fixtures, never selected from data.
    return replace(SurfaceScoringConfigV1(.5, .5, .5, .5, .5, .6, .8, .8, .8), **changes)


def test_no_loss_call_and_all32_64_queries_preserved(monkeypatch):
    import mtare_topo.representation.gse_surface_losses_v1 as loss_module
    def forbidden(*args, **kwargs):
        raise AssertionError("scorer must not consume training assignments")
    monkeypatch.setattr(loss_module, "_assign", forbidden)
    monkeypatch.setattr(loss_module, "surface_relation_losses", forbidden)
    result = score_surface_predictions(prediction(), targets(), config())
    assert result["anchor"]["f1"] == result["opening"]["f1"] == 1.
    assert result["anchor"]["counts"]["raw_queries"] == 32
    assert result["opening"]["counts"]["raw_queries"] == 64
    assert len(result["raw_predictions"]["anchor_position_m"][0]) == 32
    assert len(result["raw_predictions"]["opening_position_m"][0]) == 64
    assert result["anchor"]["counts"]["unknown_background_declared"] == 30
    assert result["opening"]["counts"]["unknown_background_declared"] == 63
    assert not result["scientific_gate_pass"]
    json.dumps(result, allow_nan=False)


def test_complete_background_surplus_is_fp_but_outside_remains_unknown():
    p, t = prediction(), targets()
    t = replace(t, anchor_region_complete=torch.ones(1, dtype=torch.bool), opening_region_complete=torch.ones(1, dtype=torch.bool))
    result = score_surface_predictions(p, t, config())
    assert result["anchor"]["fp"] == 9 and result["anchor"]["fn"] == 0
    assert result["opening"]["fp"] == 9 and result["opening"]["fn"] == 0
    assert result["anchor"]["f1"] == pytest.approx(4 / 13)
    assert result["opening"]["f1"] == pytest.approx(2 / 11)
    assert result["anchor"]["counts"]["unknown_background_declared"] == 21


def test_matching_is_not_selected_by_presence_class_or_membership():
    p, t = prediction(), targets()
    before = score_surface_predictions(p, t, config())
    presence = p.opening_presence_logits.detach().clone(); presence[0, 1] = -100.
    reaches = p.reachability_logits.detach().clone(); reaches[..., 0] = 100.
    p = replace(p, opening_presence_logits=presence, reachability_logits=reaches,
        membership_logits=torch.full_like(p.membership_logits, -100.))
    after = score_surface_predictions(p, t, config())
    a, b = before["opening"]["rows"][0]["targets"][0], after["opening"]["rows"][0]["targets"][0]
    assert a["unique_query_index"] == b["unique_query_index"] == 1
    assert after["opening"]["f1"] == 0.
    assert after["opening"]["counts"]["matched_low_presence"] == 1
    assert after["opening"]["counts"]["unknown_background_declared"] == 63
    assert after["opening"]["oracle_unique_candidate_position_error_m"]["mean"] == pytest.approx(.1)
    assert after["opening"]["detected_position_error_m"]["count"] == 0


def test_geometry_match_is_not_detection_and_tolerances_are_explicit():
    p, t = prediction(), targets()
    tight = score_surface_predictions(p, t, config(anchor_position_tolerance_m=.15, opening_position_tolerance_m=.05))
    assert tight["anchor"]["counts"]["confirmed_tp"] == 1
    assert tight["anchor"]["fn"] == 1 and tight["opening"]["fn"] == 1
    assert tight["opening"]["f1"] == 0.
    for tolerance in (1., 2., 4.):
        other = score_surface_predictions(p, t, config(anchor_position_tolerance_m=tolerance, opening_position_tolerance_m=tolerance))
        assert other["config"]["anchor_position_tolerance_m"] == tolerance
        assert other["anchor"]["counts"]["confirmed_tp"] == 2


@pytest.mark.parametrize("kind", ("duplicate_queries", "global_tie", "duplicate_targets"))
def test_geometric_ambiguity_invalidates_main_scores_not_favorable_slot(kind):
    p, t = prediction(), targets()
    if kind == "duplicate_queries":
        pos = p.anchor_position_m.detach().clone(); pos[0, 1] = pos[0, 0]
        p = replace(p, anchor_position_m=pos)
    elif kind == "duplicate_targets":
        t = replace(t, anchor_position_m=torch.zeros_like(t.anchor_position_m))
    else:
        pos = p.anchor_position_m.detach().clone(); pos[..., 0] += 2.
        p = replace(p, anchor_position_m=pos)
        t = replace(t, anchor_position_m=torch.tensor([[[0., 0., 0.], [1., 0., 0.]]], dtype=torch.float64))
    result = score_surface_predictions(p, t, config())
    assert not result["valid"] and not result["anchor"]["valid"]
    assert result["anchor"]["f1"] is result["anchor"]["precision"] is result["anchor"]["recall"] is None
    assert result["anchor"]["counts"]["ambiguous_targets"] > 0
    assert result["attributes"]["membership_known_only_conditional_f1"] is None
    for record in result["anchor"]["rows"][0]["targets"]:
        if record["ambiguous"]: assert record["unique_query_index"] is None


def test_attributes_have_separate_population_and_conditional_denominators():
    p, t = prediction(), targets()
    result = score_surface_predictions(p, t, config())["attributes"]
    assert result["direction_error_deg"] == {"count": 1, "mean": 90., "median": 90., "maximum": 90.}
    assert result["width_error_m"]["mean"] == result["height_error_m"]["mean"] == 1.
    assert result["denominators"]["membership_tp"] == 2
    assert result["denominators"]["membership_known"] == result["denominators"]["membership_scored"] == 2
    assert result["membership_full_known_population_f1_bounds"] == [1., 1.]
    assert result["reachability_confusion"] == [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]


def test_reject_all_is_not_success_via_empty_conditional_population():
    p = prediction()
    p = replace(p, anchor_presence_logits=torch.full_like(p.anchor_presence_logits, -100.),
        opening_presence_logits=torch.full_like(p.opening_presence_logits, -100.))
    result = score_surface_predictions(p, targets(), config())
    assert result["anchor"]["f1"] == result["opening"]["f1"] == 0.
    attributes = result["attributes"]
    assert attributes["membership_known_only_conditional_f1"] is None
    assert attributes["membership_full_known_population_f1_bounds"] == [0., 1.]
    assert attributes["coverage"]["membership"] == 0.
    assert attributes["denominators"]["membership_known"] == 2
    assert attributes["membership_unscored_positive"] == 2


def test_unknown_attributes_are_not_negative_and_determinate_claims_are_separate():
    p, t = prediction(), targets()
    reach = torch.full_like(p.reachability_logits, -100.); reach[..., 0] = 100.
    p = replace(p, reachability_logits=reach, membership_validity_logits=torch.full_like(p.membership_logits, 100.),
        opening_dimension_evidence_logits=torch.full_like(p.opening_dimension_evidence_logits, 100.))
    t = replace(t, direction_valid=torch.zeros_like(t.direction_valid), dimension_valid=torch.zeros_like(t.dimension_valid),
        reachability_valid=torch.zeros_like(t.reachability_valid), physical_reference_valid=torch.zeros_like(t.physical_reference_valid),
        membership_valid=torch.zeros_like(t.membership_valid), membership=torch.full_like(t.membership, float("nan")))
    result = score_surface_predictions(p, t, config())["attributes"]
    ledger = result["denominators"]
    assert ledger["membership_known"] == ledger["membership_fp"] == ledger["membership_fn"] == 0
    assert ledger["unknown_membership_reference_claims"] == 2
    assert ledger["unknown_reachability_reference_determinate_claims"] == 1
    assert ledger["unknown_direction_reference_claims"] == 1 and ledger["unknown_dimension_reference_claims"] == 2
    assert ledger["declared_opening_queries_without_detected_target"] == 63
    assert ledger["unbound_opening_determinate_reachability_claims"] == 63
    assert ledger["unbound_opening_dimension_claims"] == 126
    assert ledger["unbound_opening_membership_claims"] == 63 * 32
    assert result["membership_known_only_conditional_f1"] is None


def test_explicit_unknown_physical_reference_does_not_become_negative_class():
    p, t = prediction(), targets()
    reach = p.reachability_logits.detach().clone(); reach[..., 1] = 100.
    result = score_surface_predictions(replace(p, reachability_logits=reach),
        replace(t, reachability_class=torch.full_like(t.reachability_class, 2)), config())["attributes"]
    assert result["reachability_confusion"][2] == [0, 1, 0, 0]
    assert result["denominators"]["explicit_unknown_reachability_determinate_claims"] == 1
    assert result["unknown_claims_are_not_false_positive_labels"]


def test_untrained_reliability_heads_never_gate_primary_detection_or_known_attributes():
    p, t = prediction(), targets(); base = score_surface_predictions(p, t, config())
    other = replace(p, anchor_uncertainty_m=torch.full_like(p.anchor_uncertainty_m, 1000.),
        opening_support_logits=torch.full_like(p.opening_support_logits, -100.),
        opening_dimension_evidence_logits=torch.full_like(p.opening_dimension_evidence_logits, -100.),
        membership_validity_logits=torch.full_like(p.membership_validity_logits, -100.))
    changed = score_surface_predictions(other, t, config())
    for key in ("anchor", "opening", "attributes"):
        assert base[key] == changed[key]


def test_unknown_empty_foreground_and_complete_empty_foreground_differ():
    p, t = prediction(), targets()
    masks = {f.name: torch.zeros_like(getattr(t, f.name)) for f in fields(t) if getattr(t, f.name).dtype == torch.bool}
    t = replace(t, **masks)
    unknown = score_surface_predictions(p, t, config())
    assert unknown["anchor"]["f1"] is None and unknown["anchor"]["fp"] == 0
    complete = score_surface_predictions(p, replace(t, anchor_region_complete=torch.ones(1, dtype=torch.bool)), config())
    assert complete["anchor"]["fp"] == 11 and complete["anchor"]["f1"] == 0.


def test_query_permutation_and_common3d_rotation_preserve_scores():
    p, t = prediction(), targets()
    baseline = score_surface_predictions(p, t, config())
    a, o = torch.arange(31, -1, -1), torch.arange(63, -1, -1)
    changes = {}
    for f in fields(p):
        value = getattr(p, f.name)
        if f.name.startswith("anchor_"): changes[f.name] = value[:, a]
        elif f.name.startswith("opening_") or f.name == "reachability_logits": changes[f.name] = value[:, o]
        elif f.name.startswith("membership"): changes[f.name] = value[:, o][:, :, a]
    permuted = score_surface_predictions(replace(p, **changes), t, config())
    for key in ("anchor", "opening"):
        assert baseline[key]["f1"] == permuted[key]["f1"]
        assert baseline[key]["counts"] == permuted[key]["counts"]
    r = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
    pc = {name: getattr(p, name) @ r.T for name in ("anchor_position_m", "opening_position_m", "opening_direction")}
    tc = {name: getattr(t, name) @ r.T for name in ("anchor_position_m", "opening_position_m", "opening_direction", "score_region_center_m")}
    rotated = score_surface_predictions(replace(p, **pc), replace(t, **tc), config())
    assert rotated["attributes"] == baseline["attributes"]
    assert rotated["anchor"] == baseline["anchor"] and rotated["opening"] == baseline["opening"]


@pytest.mark.parametrize("value", (True, float("nan"), float("inf"), -1., 1.1))
def test_bad_config_has_no_silent_threshold_fallback(value):
    with pytest.raises(ValueError): config(membership_threshold=value)


def test_no_mutation_or_grad_and_explicit_config_required():
    p, t = prediction(), targets()
    tensors = [getattr(p, f.name) for f in fields(p)] + [getattr(t, f.name) for f in fields(t)]
    before = [x.detach().clone() for x in tensors]
    score_surface_predictions(p, t, config())
    for old, value in zip(before, tensors):
        torch.testing.assert_close(old, value)
        assert value.grad is None
    with pytest.raises(ValueError): score_surface_predictions(p, t, None)


def test_invalid_predicted_geometric_attributes_remain_in_known_denominator():
    p = prediction()
    p = replace(p, opening_direction=p.opening_direction * 2., opening_dimensions_m=-p.opening_dimensions_m)
    result = score_surface_predictions(p, targets(), config())["attributes"]
    assert result["direction_error_deg"]["count"] == 0
    assert result["denominators"]["direction_known"] == result["denominators"]["direction_invalid_prediction"] == 1
    assert result["denominators"]["dimensions_invalid_prediction"] == 2
    assert result["coverage"]["direction"] == result["coverage"]["width"] == result["coverage"]["height"] == 0.


def test_extreme_geometric_distance_is_rejected_not_hidden_as_outside_region():
    p = prediction()
    p = replace(p, anchor_position_m=torch.full_like(p.anchor_position_m, 1e308))
    with pytest.raises(ValueError, match="overflow"):
        score_surface_predictions(p, targets(), config())
