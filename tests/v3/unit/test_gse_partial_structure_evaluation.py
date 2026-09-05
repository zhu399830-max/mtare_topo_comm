"""Synthetic capacity accounting; no datasets, weights, or training runs."""
from dataclasses import replace
import inspect

import pytest
import torch

from mtare_topo.evaluation.gse_partial_structure import evaluate_partial_structure
from mtare_topo.representation.gse_region_queries import RegionPrediction, RegionTargets


def fixture(n=4, m=2):
    centers = torch.zeros(1, n, 3, dtype=torch.float64)
    centers[0, :, 0] = torch.arange(n) * 10
    support = torch.ones(1, n, dtype=torch.bool)
    event_logits = torch.full((1, n, 3), -3., dtype=centers.dtype)
    event_logits[:, :, 0] = 3
    if n > 1: event_logits[0, 1] = torch.tensor([-3., 3., -3.])
    logits = torch.full((1, n, n), -3., dtype=centers.dtype)
    for i in range(n): logits[0, i, i] = 3
    p = RegionPrediction(centers, event_logits, torch.zeros(1, n, dtype=centers.dtype), logits,
                         torch.zeros(1, n, dtype=centers.dtype), support, support[:, :, None] & support[:, None])
    targets = torch.zeros(1, m, 3, dtype=centers.dtype)
    targets[0, :, 0] = torch.arange(m) * 10
    events = torch.arange(m)[None] % 2
    members = torch.zeros(1, m, n, dtype=centers.dtype)
    valid = torch.zeros(1, m, n, dtype=torch.bool)
    for i in range(m):
        if i < n: members[0, i, i] = 1
        valid[0, i, :min(n, 2)] = True
    t = RegionTargets(targets, torch.ones(1, m, dtype=torch.bool), events, torch.ones(1, m, dtype=torch.bool),
                      members, valid, torch.zeros(1, dtype=torch.bool))
    return p, t


def evaluate(p, t, **kwargs):
    return evaluate_partial_structure(p, t, membership_threshold=.5, **kwargs)


def test_exact_capacity_counts_are_not_detection():
    p, t = fixture()
    result = evaluate(p, t)
    assert result.unique_center_query.tolist() == [[0, 1]]
    assert result.summary["geometry"]["unselected_queries"] == 2
    assert result.summary["events"]["two_class_macro_f1"] == 1
    assert result.summary["members"]["known_scored_f1"] == 1
    assert result.summary["members"]["coverage"] == 1
    assert result.summary["unique_assigned_distance"]["mean_m"] == 0
    assert not result.summary["complete_detection_claim"]
    assert not result.summary["unmatched_queries_are_false_positives"]


def test_event_members_presence_uncertainty_cannot_choose_correspondence():
    p, t = fixture()
    before = evaluate(p, t)
    event = p.event_logits.clone(); event[:, :2] = event[:, :2].flip(1)
    membership = -p.membership_logits
    changed = replace(p, event_logits=event, membership_logits=membership,
                      presence_logits=torch.randn_like(p.presence_logits)*100,
                      uncertainty=torch.rand_like(p.uncertainty))
    after = evaluate(changed, t)
    assert torch.equal(before.unique_center_query, after.unique_center_query)
    assert after.summary["events"]["two_class_macro_f1"] == 0
    assert after.summary["members"]["known_scored_f1"] == 0


def test_duplicate_center_rejects_semantically_perfect_candidate():
    p, t = fixture()
    p.centers_m[0, 3] = p.centers_m[0, 0]
    p.event_logits[0, 3] = torch.tensor([-3., 3., -3.])
    result = evaluate(p, t)
    assert result.unique_center_query.tolist() == [[-1, 1]]
    assert result.summary["geometry"]["ambiguous_pairs"] == 1
    assert result.summary["events"]["confusion"] == [[0, 0, 0, 1], [0, 1, 0, 0]]
    members = result.summary["members"]
    assert members["rejected"]["center_ambiguous"] == dict(positive=1, negative=1)
    assert members["known_scored_f1"] == 1
    assert members["worst_case_f1"] == .5
    assert members["best_case_f1"] == 1
    assert members["coverage"] == .5


def test_all_rejected_never_f1_one_or_zero_loss_success():
    p, t = fixture()
    p.query_supported[:] = False; p.member_supported[:] = False
    result = evaluate(p, t)
    assert result.summary["members"]["known_scored_f1"] is None
    assert result.summary["members"]["coverage"] == 0
    assert result.summary["members"]["worst_case_f1"] == 0
    assert result.summary["members"]["best_case_f1"] == 1
    assert result.summary["events"]["two_class_macro_f1"] == 0
    assert result.summary["assigned_distance_representative"]["mean_m"] is None
    assert result.summary["geometry"]["unselected_centers"] == 2


def test_prediction_terminal_counts_as_wrong_not_removed():
    p, t = fixture()
    p.event_logits[0, 1] = torch.tensor([-2., -2., 4.])
    result = evaluate(p, t)
    assert result.summary["events"]["confusion"][1] == [0, 0, 1, 0]
    assert result.summary["events"]["per_class"]["junction"]["recall"] == 0


def test_unknown_event_excluded_and_class_tie_rejected():
    p, t = fixture()
    t.event_valid[0, 0] = False; t.events[0, 0] = -1
    p.event_logits[0, 1] = 0
    result = evaluate(p, t)
    assert result.summary["events"]["known_events"] == 1
    assert result.summary["events"]["confusion"][1] == [0, 0, 0, 1]
    assert result.summary["events"]["two_class_macro_f1"] is None


def test_always_corridor_baseline_exposes_imbalance():
    p, t = fixture(n=5, m=5)
    t.events[:] = 0; t.events[0, 4] = 1
    p.event_logits[:] = torch.tensor([3., -2., -2.])
    result = evaluate(p, t)
    assert result.summary["always_corridor_baseline"]["accuracy_diagnostic_only"] == .8
    assert result.summary["events"]["confusion"] == result.summary["always_corridor_baseline"]["confusion"]
    assert result.summary["events"]["per_class"]["junction"]["f1"] == 0


def test_unsupported_direction_accounted_not_negative():
    p, t = fixture()
    t.member_valid[0, 0, 3] = True; t.members[0, 0, 3] = 1
    p.query_supported[0, 3] = False
    p.member_supported[:] = p.query_supported[:, :, None] & p.query_supported[:, None]
    result = evaluate(p, t)
    assert result.summary["members"]["rejected"]["unsupported_direction"] == dict(positive=1, negative=0)
    assert not result.scored_member_mask[0, 0, 3]
    assert result.summary["members"]["positive_coverage"] == 2/3


def test_prebridge_population_and_worst_best_bounds():
    p, t = fixture()
    ledger = [dict(original_member_positive=5, original_member_negative=6,
                   transferred_member_positive=2, transferred_member_negative=2,
                   unknown_correspondence_member_positive=3, unknown_correspondence_member_negative=4)]
    result = evaluate(p, t, direction_bridge_ledger=ledger)
    members = result.summary["members"]
    assert members["coverage"] == 4/11
    assert members["worst_case_f1"] == 4/11
    assert members["best_case_f1"] == 1
    assert members["bound_population"] == "PRE_BRIDGE_KNOWN_LABELS"


def test_bad_bridge_conservation_rejected():
    p, t = fixture()
    ledger = [dict(original_member_positive=2, original_member_negative=2,
                   transferred_member_positive=1, transferred_member_negative=2,
                   unknown_correspondence_member_positive=1, unknown_correspondence_member_negative=0)]
    with pytest.raises(ValueError, match="conservation"):
        evaluate(p, t, direction_bridge_ledger=ledger)


def test_ambiguous_optimum_not_just_duplicate_points():
    p, t = fixture(n=2)
    p.centers_m[0] = torch.tensor([[-1., 0., 0.], [-2., 0., 0.]])
    result = evaluate(p, t)
    assert result.summary["geometry"]["ambiguous_pairs"] == 2
    assert not result.scored_member_mask.any()


def test_one_ulp_geometry_is_unknown():
    p, t = fixture()
    p.centers_m[0, 3] = p.centers_m[0, 1]
    p.centers_m[0, 3, 0] = torch.nextafter(p.centers_m[0, 3, 0], torch.tensor(float("inf")))
    result = evaluate(p, t)
    assert result.unique_center_query[0, 1] == -1


def test_noisy_rotations_and_query_permutation_preserve_scores():
    p, t = fixture()
    p.centers_m[0, :2] += torch.tensor([[.4, .2, .3], [-.3, .2, -.1]])
    q, _ = torch.linalg.qr(torch.tensor([[1., 2., 3.], [3., 1., 4.], [2., 5., 1.]], dtype=torch.float64))
    first = evaluate(p, t)
    rotated = evaluate(replace(p, centers_m=p.centers_m @ q.T), replace(t, centers_m=t.centers_m @ q.T))
    assert torch.equal(first.unique_center_query, rotated.unique_center_query)
    assert first.summary["events"] == rotated.summary["events"]
    perm = torch.tensor([2, 0, 3, 1])
    pp = replace(p, centers_m=p.centers_m[:, perm], event_logits=p.event_logits[:, perm],
                 membership_logits=p.membership_logits[:, perm][:, :, perm], presence_logits=p.presence_logits[:, perm],
                 uncertainty=p.uncertainty[:, perm], query_supported=p.query_supported[:, perm],
                 member_supported=p.member_supported[:, perm][:, :, perm])
    tt = replace(t, members=t.members[:, :, perm], member_valid=t.member_valid[:, :, perm])
    after = evaluate(pp, tt)
    assert first.summary["events"] == after.summary["events"]
    assert first.summary["members"] == after.summary["members"]


def test_fewer_queries_and_missing_center_have_separate_rejections():
    p, t = fixture(n=2, m=3)
    t.center_valid[0, 2] = False
    result = evaluate(p, t)
    assert result.summary["members"]["rejected"]["center_missing"]["negative"] == 2
    t.center_valid[0, 2] = True
    result = evaluate(p, t)
    assert result.summary["geometry"]["unselected_centers"] == 1
    assert result.summary["members"]["rejected"]["center_unselected_or_alternative"]["negative"] == 2


def test_no_targets_64_candidates_not_false_positives():
    p, t = fixture(n=64, m=0)
    result = evaluate(p, t)
    assert result.summary["geometry"]["unselected_queries"] == 64
    assert result.summary["events"]["two_class_macro_f1"] is None
    assert result.summary["members"]["coverage"] is None
    assert result.summary["members"]["known_scored_f1"] is None


def test_64_candidates_13_targets_is_capacity_only():
    p, t = fixture(n=64, m=13)
    result = evaluate(p, t)
    assert result.summary["geometry"]["unique_pairs"] == 13
    assert result.summary["geometry"]["unselected_queries"] == 51
    assert not result.summary["complete_detection_claim"]


def test_metadata_does_not_choose_matches():
    p, t = fixture()
    a = evaluate(p, t, manifest=[dict(parent="one", target_identity=1)])
    b = evaluate(p, t, manifest=[dict(parent="two", target_identity=999)])
    assert a.summary == b.summary
    assert torch.equal(a.unique_center_query, b.unique_center_query)


@pytest.mark.parametrize("threshold", [0., .4, 1., True, float("nan")])
def test_threshold_not_tunable(threshold):
    with pytest.raises(ValueError, match="0.5"):
        evaluate_partial_structure(*fixture(), membership_threshold=threshold)


@pytest.mark.parametrize("failure", ["nan", "complete", "terminal_teacher", "soft_member", "support", "overflow"])
def test_invalid_inputs(failure):
    p, t = fixture()
    if failure == "nan": p.centers_m[0, 0, 0] = float("nan")
    if failure == "complete": t.label_complete[:] = True
    if failure == "terminal_teacher": t.events[0, 0] = 2
    if failure == "soft_member": t.members[0, 0, 0] = .2
    if failure == "support": p.member_supported[0, 0, 1] = False
    if failure == "overflow": p.centers_m[0, 0, 0] = 1e200
    with pytest.raises(ValueError): evaluate(p, t)


def test_api_has_no_training_matches_and_does_not_mutate_tensors():
    p, t = fixture()
    before = p.centers_m.clone(), t.members.clone()
    result = evaluate(p, t)
    assert "matches" not in inspect.signature(evaluate_partial_structure).parameters
    assert torch.equal(before[0], p.centers_m) and torch.equal(before[1], t.members)
    assert not result.scored_member_mask.requires_grad
