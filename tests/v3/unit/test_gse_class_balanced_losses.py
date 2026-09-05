"""Synthetic 2x2 loss/gradient controls; no datasets, models or training."""
from dataclasses import fields
import math

import pytest
import torch

import mtare_topo.representation.gse_class_balanced_losses as balanced
from mtare_topo.representation.gse_geometry_bound_losses import geometry_bound_region_losses
from tests.v3.unit.test_gse_geometry_bound_losses import prediction, target


def fixture():
    p = prediction([[.1, 0., 0.], [10.1, 0., 0.]])
    t = target([[0., 0., 0.], [10., 0., 0.]], 2)
    t.events[0] = torch.tensor([0, 1])
    t.members[0] = torch.tensor([[1., 0.], [0., 1.]])
    return p, t


def test_00_returns_exact_original_dictionary_object(monkeypatch):
    sentinel = {"untouched": object()}
    monkeypatch.setattr(balanced, "geometry_bound_region_losses", lambda *args: sentinel)
    assert balanced.class_balanced_region_losses(None, None) is sentinel


def test_00_real_losses_counts_matches_and_gradients_exact():
    p, t = fixture()
    original = geometry_bound_region_losses(p, t)
    result = balanced.class_balanced_region_losses(p, t)
    assert set(original) == set(result)
    for key, value in original.items():
        if isinstance(value, torch.Tensor): assert torch.equal(value, result[key])
        else: assert value == result[key]
    params = tuple(getattr(p, key) for key in ("centers_m", "event_logits", "membership_logits", "presence_logits", "uncertainty"))
    a = torch.autograd.grad(original["total"], params, retain_graph=True)
    b = torch.autograd.grad(result["total"], params)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("members,events", [(True, False), (False, True), (True, True)])
def test_2x2_only_requested_task_changes_and_global_fixed_denominators(members, events):
    p, t = fixture()
    original = geometry_bound_region_losses(p, t)
    result = balanced.class_balanced_region_losses(p, t,
        member_class_counts=(9, 1) if members else None, event_class_counts=(9, 1, 0) if events else None)
    mean_global_weight_on_this_balanced_batch = (10/18 + 10/2)/2
    assert float(result["membership"].detach()) == pytest.approx(math.log(2) * (mean_global_weight_on_this_balanced_batch if members else 1))
    assert float(result["event"].detach()) == pytest.approx(math.log(3) * (mean_global_weight_on_this_balanced_batch if events else 1))
    for key in ("center", "presence", "uncertainty"):
        assert result[key] is original[key] or torch.equal(result[key], original[key])
    for key in ("counts", "matches", "geometry_counts", "geometry_matches", "supervision_counts", "denominators", "has_supervision"):
        assert result[key] == original[key]
    assert torch.equal(result["unique_center_query"], original["unique_center_query"])
    assert result["denominators"]["membership"] == 4 and result["denominators"]["event"] == 2


def test_member_and_event_gradients_are_only_globally_scaled():
    p, t = fixture()
    original = geometry_bound_region_losses(p, t)
    result = balanced.class_balanced_region_losses(p, t, member_class_counts=(9, 1), event_class_counts=(9, 1, 0))
    names = ("centers_m", "event_logits", "membership_logits", "presence_logits", "uncertainty")
    params = tuple(getattr(p, key) for key in names)
    old = dict(zip(names, torch.autograd.grad(original["total"], params, retain_graph=True)))
    new = dict(zip(names, torch.autograd.grad(result["total"], params)))
    for key in ("centers_m", "presence_logits", "uncertainty"):
        assert torch.equal(new[key], old[key])
    weight = torch.tensor([10/18, 5.], dtype=p.centers_m.dtype)
    torch.testing.assert_close(new["membership_logits"], old["membership_logits"] * weight[t.members.long()], rtol=0, atol=1e-15)
    torch.testing.assert_close(new["event_logits"], old["event_logits"] * weight[t.events][..., None], rtol=0, atol=1e-15)
    # Terminal retains a softmax gradient despite no terminal-positive labels.
    assert bool((new["event_logits"][..., 2] > 0).all())


def test_uniform_global_counts_recover_unweighted_tasks():
    p, t = fixture()
    a = geometry_bound_region_losses(p, t)
    b = balanced.class_balanced_region_losses(p, t, member_class_counts=(20, 20), event_class_counts=(10, 10, 0))
    for key in ("center", "event", "membership", "presence", "uncertainty", "total"):
        assert torch.equal(a[key], b[key])


def test_class_missing_from_local_batch_does_not_recompute_weights():
    p, t = fixture()
    t.events[:] = 1; t.members[:] = 1
    result = balanced.class_balanced_region_losses(p, t, member_class_counts=(13489, 2152), event_class_counts=(872, 14, 0))
    assert float(result["membership"].detach()) == pytest.approx(math.log(2) * 15641/(2*2152))
    assert float(result["event"].detach()) == pytest.approx(math.log(3) * 886/(2*14))


def test_ambiguous_geometry_preserves_unknown_and_zero_supervision():
    p, t = fixture()
    with torch.no_grad(): p.centers_m[0, 1].copy_(p.centers_m[0, 0])
    original = geometry_bound_region_losses(p, t)
    result = balanced.class_balanced_region_losses(p, t, member_class_counts=(9, 1), event_class_counts=(9, 1, 0))
    assert not result["has_supervision"]
    assert result["counts"] == original["counts"]
    assert result["supervision_counts"] == original["supervision_counts"]
    assert float(result["total"].detach()) == 0
    assert result["denominators"] == original["denominators"]


def test_unknown_label_never_gets_weight_or_gradient():
    p, t = fixture()
    t.member_valid[0, 0, 1] = False; t.members[0, 0, 1] = float("nan")
    t.event_valid[0, 0] = False; t.events[0, 0] = -1
    result = balanced.class_balanced_region_losses(p, t, member_class_counts=(9, 1), event_class_counts=(9, 1, 0))
    grads = torch.autograd.grad(result["membership"] + result["event"], (p.membership_logits, p.event_logits))
    assert grads[0][0, 0, 1] == 0
    assert bool((grads[1][0, 0] == 0).all())
    assert result["counts"]["membership"] == 3 and result["counts"]["event"] == 1


@pytest.mark.parametrize("member_counts,event_counts", [((0, 0), None), ((0, 10), None), ((1, -1), None),
    ((True, 5), None), ((1., 5), None), ([1, 5], None), ((1, 5, 6), None),
    (None, (0, 0, 0)), (None, (10, 0, 0)), (None, (1, float("inf"), 0)),
    (None, (10, 5)), (None, (1, 10**400, 0))])
def test_illegal_or_insufficient_global_class_counts_rejected(member_counts, event_counts):
    with pytest.raises(ValueError):
        balanced.class_balanced_region_losses(*fixture(), member_class_counts=member_counts, event_class_counts=event_counts)


def test_known_terminal_with_global_zero_count_rejected_even_if_geometry_ambiguous():
    p, t = fixture(); t.events[0, 0] = 2
    with torch.no_grad(): p.centers_m[0, 1].copy_(p.centers_m[0, 0])
    with pytest.raises(ValueError, match="zero global support"):
        balanced.class_balanced_region_losses(p, t, event_class_counts=(872, 14, 0))


def test_known_corridor_zero_count_with_other_two_supported_classes_rejected():
    with pytest.raises(ValueError, match="zero global support"):
        balanced.class_balanced_region_losses(*fixture(), event_class_counts=(0, 3, 4))


def test_soft_member_targets_rejected_only_when_member_balance_enabled():
    p, t = fixture(); t.members[0, 0, 0] = .4
    balanced.class_balanced_region_losses(p, t)
    with pytest.raises(ValueError, match="binary"):
        balanced.class_balanced_region_losses(p, t, member_class_counts=(9, 1))


def test_prediction_target_inputs_unmodified():
    p, t = fixture()
    snapshots = [{f.name: getattr(value, f.name).detach().clone() for f in fields(value)} for value in (p, t)]
    balanced.class_balanced_region_losses(p, t, member_class_counts=(9, 1), event_class_counts=(9, 1, 0))
    for value, snapshot in zip((p, t), snapshots):
        assert all(torch.equal(getattr(value, name), tensor) for name, tensor in snapshot.items())
