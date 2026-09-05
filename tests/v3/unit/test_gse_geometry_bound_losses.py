"""Synthetic loss binding only; no model, optimizer, or real observations."""
from dataclasses import fields, replace

import pytest
import torch
from torch.nn import functional as F

from mtare_topo.evaluation.gse_partial_structure import evaluate_partial_structure
from mtare_topo.representation.gse_geometry_bound_losses import geometry_bound_region_losses
from mtare_topo.representation.gse_region_queries import RegionPrediction, RegionTargets, region_set_losses


def prediction(centers):
    c = torch.tensor(centers, dtype=torch.float64)
    if c.ndim == 2: c = c[None]
    b, n = c.shape[:2]
    q = torch.ones(b, n, dtype=torch.bool)
    def leaf(v): return v.clone().requires_grad_()
    return RegionPrediction(leaf(c), leaf(torch.zeros(b, n, 3, dtype=c.dtype)),
        leaf(torch.zeros(b, n, dtype=c.dtype)), leaf(torch.zeros(b, n, n, dtype=c.dtype)),
        leaf(torch.full((b, n), .3, dtype=c.dtype)), q, q[:, :, None] & q[:, None])


def target(centers, n):
    c = torch.tensor(centers, dtype=torch.float64)
    if c.ndim == 2: c = c[None]
    b, m = c.shape[:2]
    return RegionTargets(c, torch.ones(b, m, dtype=torch.bool), torch.zeros(b, m, dtype=torch.long),
        torch.ones(b, m, dtype=torch.bool), torch.ones(b, m, n, dtype=c.dtype),
        torch.ones(b, m, n, dtype=torch.bool), torch.zeros(b, dtype=torch.bool))


def test_far_class_good_candidate_cannot_steal_near_center_target():
    p = prediction([[.2, 0, 0], [60, 0, 0]])
    t = target([[0, 0, 0]], 2)
    with torch.no_grad():
        p.event_logits[0] = torch.tensor([[-10., 10., -10.], [10., -10., -10.]])
    historical = region_set_losses(p, t)
    bound = geometry_bound_region_losses(p, t)
    assert historical["matches"] == [(0, 1, 0)]
    assert bound["matches"] == [(0, 0, 0)]
    assert bound["event"] > historical["event"]
    assert bound["center"].item() == pytest.approx(.2 / 50)


def test_target_class_members_and_prediction_semantic_scores_cannot_change_assignment():
    p = prediction([[.2, 0, 0], [60, 0, 0]])
    t = target([[0, 0, 0]], 2)
    a = geometry_bound_region_losses(p, t)
    changed = replace(t, events=torch.ones_like(t.events), members=torch.zeros_like(t.members))
    with torch.no_grad():
        p.event_logits[0, 1] = torch.tensor([0., 100., 0.])
        p.membership_logits[0, 1] = -100
        p.presence_logits[0, 1] = 100
        p.uncertainty[0, 1] = 0
    b = geometry_bound_region_losses(p, changed)
    assert a["matches"] == b["matches"] == [(0, 0, 0)]
    assert torch.equal(a["unique_center_query"], b["unique_center_query"])


def test_only_geometrically_matched_query_receives_semantic_gradients():
    p = prediction([[.2, 0, 0], [60, 0, 0]])
    loss = geometry_bound_region_losses(p, target([[0, 0, 0]], 2))
    loss["total"].backward()
    for name in ("centers_m", "event_logits", "presence_logits", "membership_logits", "uncertainty"):
        grad = getattr(p, name).grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad[0, 0].abs().sum() > 0
        assert grad[0, 1].abs().sum() == 0


@pytest.mark.parametrize("centers,truth", [
    ([[1, 0, 0], [10, 0, 0], [30, 0, 0]], [[0, 0, 0], [11, 0, 0]]),
    ([[1, 0, 0], [-1, 0, 0]], [[0, 0, 0]]),
    ([[0, 0, 0]], [[-1, 0, 0], [1, 0, 0]]),
    ([[0, 0, 0], [2, 0, 0]], [[1, 0, 0], [1, 0, 0]]),
])
def test_same_unique_mapping_as_original_independent_evaluator(centers, truth):
    p, t = prediction(centers), target(truth, len(centers))
    actual = geometry_bound_region_losses(p, t)
    expected = evaluate_partial_structure(p, t, membership_threshold=.5)
    assert torch.equal(actual["unique_center_query"].cpu(), expected.unique_center_query)


def test_exact_tie_has_no_semantic_supervision_or_index_winner():
    p = prediction([[-1, 0, 0], [1, 0, 0]])
    t = target([[0, 0, 0]], 2)
    with torch.no_grad(): p.event_logits[0, 1, 0] = 100
    result = geometry_bound_region_losses(p, t)
    assert result["matches"] == [] and not result["has_supervision"]
    assert result["geometry_counts"]["ambiguous_pairs"] == 1
    assert result["counts"]["unmatched_targets"] == 1
    assert result["supervision_counts"]["unbound_event_labels"] == 1
    assert result["supervision_counts"]["unbound_usable_member_labels"] == 2
    assert result["total"].item() == 0
    result["total"].backward()
    assert p.event_logits.grad.count_nonzero() == 0


def test_numerical_near_tie_matches_original_roundoff_and_semantics_cannot_break_it():
    p = prediction([[-1, 0, 0], [1 + 1e-15, 0, 0]])
    t = target([[0, 0, 0]], 2)
    assert not geometry_bound_region_losses(p, t)["has_supervision"]
    assert evaluate_partial_structure(p, t, membership_threshold=.5).unique_center_query[0, 0] == -1


def test_center_only_keeps_original_presence_and_center_scale():
    p = prediction([[2, 0, 0]])
    t = target([[0, 0, 0]], 1)
    t = replace(t, event_valid=torch.zeros_like(t.event_valid), member_valid=torch.zeros_like(t.member_valid))
    old, new = region_set_losses(p, t), geometry_bound_region_losses(p, t)
    for key in ("center", "event", "membership", "presence", "uncertainty", "total"):
        torch.testing.assert_close(old[key], new[key])
    assert new["center"].item() == pytest.approx(2 / 50)
    assert new["presence"].item() == pytest.approx(float(F.softplus(torch.tensor(0.))))


def test_empty_known_population_returns_explicit_no_supervision():
    p = prediction([[1, 0, 0], [2, 0, 0]])
    t = target([[0, 0, 0]], 2)
    t = replace(t, centers_m=torch.full_like(t.centers_m, float("nan")),
        center_valid=torch.zeros_like(t.center_valid), event_valid=torch.zeros_like(t.event_valid),
        events=torch.full_like(t.events, -1), member_valid=torch.zeros_like(t.member_valid),
        members=torch.full_like(t.members, float("nan")))
    loss = geometry_bound_region_losses(p, t)
    assert not loss["has_supervision"] and loss["counts"]["targets"] == 0
    assert loss["total"].item() == 0
    loss["total"].backward()
    assert torch.isfinite(p.centers_m.grad).all()


def test_finite_large_unused_predictions_do_not_make_unknown_zero_nan():
    p = prediction([[1, 0, 0], [2, 0, 0]])
    with torch.no_grad():
        p.event_logits.fill_(1e308)
        p.membership_logits.fill_(1e308)
    t = target([[0, 0, 0]], 2)
    t = replace(t, center_valid=torch.zeros_like(t.center_valid),
        event_valid=torch.zeros_like(t.event_valid), member_valid=torch.zeros_like(t.member_valid))
    loss = geometry_bound_region_losses(p, t)
    assert loss["total"].item() == 0 and not loss["has_supervision"]
    loss["total"].backward()
    assert torch.isfinite(p.event_logits.grad).all()


def test_generic_three_class_loss_does_not_depend_on_two_class_evaluator():
    p = prediction([[1, 0, 0]])
    t = target([[0, 0, 0]], 1)
    t = replace(t, events=torch.full_like(t.events, 2))
    loss = geometry_bound_region_losses(p, t)
    assert loss["event"].item() == pytest.approx(float(torch.log(torch.tensor(3.))))
    assert loss["matches"] == [(0, 0, 0)]


def test_duplicate_batch_preserves_loss_scale_and_original_denominators():
    p = prediction([[1, 0, 0]])
    t = target([[0, 0, 0]], 1)
    original = geometry_bound_region_losses(p, t)
    def doubled(value):
        return type(value)(**{f.name: torch.cat([getattr(value, f.name)] * 2) for f in fields(value)})
    p2, t2 = doubled(p), doubled(t)
    actual, old = geometry_bound_region_losses(p2, t2), region_set_losses(p2, t2)
    for key in ("center", "event", "membership", "presence", "uncertainty", "total"):
        torch.testing.assert_close(original[key], actual[key])
        torch.testing.assert_close(old[key], actual[key])
    assert actual["denominators"] == dict.fromkeys(("center", "event", "membership", "presence", "uncertainty"), 2)


def test_ambiguous_row_stays_in_batch_denominator_not_renormalized_away():
    p = prediction([[[1, 0, 0], [10, 0, 0]], [[-1, 0, 0], [1, 0, 0]]])
    t = target([[[0, 0, 0]], [[0, 0, 0]]], 2)
    both = geometry_bound_region_losses(p, t)
    def first(value): return type(value)(**{f.name: getattr(value, f.name)[:1] for f in fields(value)})
    one = geometry_bound_region_losses(first(p), first(t))
    for key in ("center", "event", "membership", "presence", "uncertainty", "total"):
        torch.testing.assert_close(both[key], one[key] / 2)
    assert both["counts"]["targets"] == 2 and both["counts"]["matched"] == 1


def test_unsupported_queries_and_members_remain_in_accounting():
    p = prediction([[1, 0, 0], [0, 0, 0]])
    q = torch.tensor([[True, False]])
    p = replace(p, query_supported=q, member_supported=q[:, :, None] & q[:, None])
    t = target([[0, 0, 0]], 2)
    loss = geometry_bound_region_losses(p, t)
    assert loss["matches"] == [(0, 0, 0)]
    assert loss["counts"]["membership"] == 1
    assert loss["supervision_counts"]["unsupported_member_labels"] == 1
    assert loss["geometry_counts"]["invalid_queries"] == 1


def test_query_permutation_preserves_unique_supervision_and_loss():
    p = prediction([[1, 0, 0], [10, 0, 0], [30, 0, 0]])
    t = target([[0, 0, 0], [11, 0, 0]], 3)
    perm = torch.tensor([2, 0, 1])
    changed = RegionPrediction(p.centers_m[:, perm], p.event_logits[:, perm], p.presence_logits[:, perm],
        p.membership_logits[:, perm][:, :, perm], p.uncertainty[:, perm], p.query_supported[:, perm],
        p.member_supported[:, perm][:, :, perm])
    moved_target = replace(t, members=t.members[:, :, perm], member_valid=t.member_valid[:, :, perm])
    a, b = geometry_bound_region_losses(p, t), geometry_bound_region_losses(changed, moved_target)
    assert {(r, int(perm[q]), t) for r, q, t in b["matches"]} == set(a["matches"])
    torch.testing.assert_close(a["total"], b["total"])


def test_all_queries_unsupported_keeps_known_targets_unbound():
    p = prediction([[1, 0, 0], [2, 0, 0]])
    valid = torch.zeros_like(p.query_supported)
    p = replace(p, query_supported=valid, member_supported=valid[:, :, None] & valid[:, None])
    t = target([[0, 0, 0]], 2)
    result = geometry_bound_region_losses(p, t)
    assert not result["has_supervision"] and result["matches"] == []
    assert result["counts"]["unmatched_targets"] == 1
    assert result["supervision_counts"]["unbound_event_labels"] == 1
    assert result["supervision_counts"]["unsupported_member_labels"] == 2
    assert result["total"].item() == 0


@pytest.mark.parametrize("kind", ["event", "member"])
def test_known_semantics_without_known_center_explicitly_refused(kind):
    p = prediction([[1, 0, 0]])
    t = target([[0, 0, 0]], 1)
    t = replace(t, center_valid=torch.zeros_like(t.center_valid),
        event_valid=torch.full_like(t.event_valid, kind == "event"),
        member_valid=torch.full_like(t.member_valid, kind == "member"))
    with pytest.raises(ValueError, match="requires known center"):
        geometry_bound_region_losses(p, t)


@pytest.mark.parametrize("issue", ["complete", "event_range", "nan_known", "nan_prediction", "mask_dtype", "member_range"])
def test_invalid_contract_refused(issue):
    p, t = prediction([[1, 0, 0]]), target([[0, 0, 0]], 1)
    if issue == "complete": t.label_complete[:] = True
    if issue == "event_range": t.events[:] = 3
    if issue == "nan_known": t.centers_m[:] = float("nan")
    if issue == "nan_prediction":
        with torch.no_grad(): p.centers_m[:] = float("nan")
    if issue == "mask_dtype": t = replace(t, center_valid=t.center_valid.int())
    if issue == "member_range": t.members[:] = 1.1
    with pytest.raises(ValueError): geometry_bound_region_losses(p, t)
