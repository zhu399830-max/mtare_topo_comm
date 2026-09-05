"""Synthetic invariance/readout contracts, zero real input or training."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from mtare_topo.representation.gse_membership_event_readout import (
    MembershipConditionedEventReadout, membership_event_features,
)
from mtare_topo.representation.gse_region_queries import AxisTokens, tokens_from_axes
from tests.v3.unit.test_gse_geometry_bound_losses import prediction


def inputs():
    axes = torch.tensor([[[[-2., 0., 1.], [-1., 0., 1.], [0., 0., 1.]],
                           [[2., 1., 0.], [2., 2., 0.], [2., 3., 0.]]]], dtype=torch.float64)
    tokens = tokens_from_axes(axes)
    p = prediction(tokens.positions_m.tolist())
    with torch.no_grad():
        p.membership_logits.copy_(torch.tensor([[[-2., -1., 1., 3.]]], dtype=torch.float64).expand(1, 4, 4))
    return p, tokens


def test_exact_features_and_parameter_count_no_input_mutation():
    p, t = inputs()
    before = deepcopy(p)
    model = MembershipConditionedEventReadout().double()
    out = model(p, t)
    assert sum(x.numel() for x in model.parameters()) == 707
    assert out.event_logits.shape == (1, 4, 3) and out.features.shape == (1, 4, 18)
    assert torch.equal(out.query_supported, t.valid)
    assert torch.equal(out.features[..., -1], torch.full((1, 4), 4., dtype=torch.float64))
    torch.testing.assert_close(out.features[..., -2], p.membership_logits.sigmoid().sum(-1))
    for field in p.__dataclass_fields__:
        assert torch.equal(getattr(p, field), getattr(before, field))


def test_no_membership_changes_only_nine_columns_and_same_weights():
    p, t = inputs()
    model = MembershipConditionedEventReadout().double()
    ablation = deepcopy(model); ablation.use_membership = False
    a, b = model(p, t), ablation(p, t)
    assert torch.equal(a.features[..., :8], b.features[..., :8])
    assert torch.equal(a.features[..., -1], b.features[..., -1])
    assert torch.count_nonzero(b.features[..., 8:17]) == 0
    assert all(torch.equal(v, ablation.state_dict()[k]) for k, v in model.state_dict().items())
    changed = replace(p, membership_logits=p.membership_logits + 10)
    assert torch.equal(ablation(p, t).event_logits, ablation(changed, t).event_logits)


def test_same_plain_geometry_different_membership_is_retained_without_threshold():
    p, t = inputs()
    a = membership_event_features(p, t)
    changed = replace(p, membership_logits=p.membership_logits.flip(-1))
    b = membership_event_features(changed, t)
    assert torch.equal(a[..., :8], b[..., :8])
    torch.testing.assert_close(a[..., -2:], b[..., -2:])
    assert not torch.allclose(a[..., 8:16], b[..., 8:16])
    changed = replace(p, membership_logits=p.membership_logits + .01)
    assert not torch.equal(a[..., -2], membership_event_features(changed, t)[..., -2])


def test_common_yaw_invariance():
    p, t = inputs()
    angle = torch.tensor(.731, dtype=torch.float64)
    c, s = angle.cos(), angle.sin()
    r = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)
    rotated = AxisTokens(t.positions_m @ r.T, t.tangents @ r.T, t.valid)
    changed = replace(p, centers_m=p.centers_m @ r.T)
    model = MembershipConditionedEventReadout().double()
    torch.testing.assert_close(model(p, t).features, model(changed, rotated).features, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(model(p, t).event_logits, model(changed, rotated).event_logits, atol=1e-12, rtol=1e-12)


def test_query_and_direction_permutation_equivariance():
    p, t = inputs(); order = torch.tensor([2, 0, 3, 1])
    changed = replace(p, **{k: (getattr(p, k)[:, order][:, :, order] if k in ("membership_logits", "member_supported")
                                else getattr(p, k)[:, order]) for k in p.__dataclass_fields__})
    tokens = AxisTokens(t.positions_m[:, order], t.tangents[:, order], t.valid[:, order])
    model = MembershipConditionedEventReadout().double()
    torch.testing.assert_close(model(p, t).features[:, order], model(changed, tokens).features)
    torch.testing.assert_close(model(p, t).event_logits[:, order], model(changed, tokens).event_logits)


def test_default_gradients_only_new_head_and_no_old_logits_in_features():
    p, t = inputs()
    t = AxisTokens(t.positions_m.clone().requires_grad_(), t.tangents.clone().requires_grad_(), t.valid)
    model = MembershipConditionedEventReadout().double()
    model(p, t).event_logits.square().sum().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())
    assert all(getattr(p, f).grad is None for f in ("centers_m", "membership_logits", "event_logits", "presence_logits", "uncertainty"))
    assert t.positions_m.grad is None and t.tangents.grad is None
    changed = replace(p, event_logits=p.event_logits + 100, presence_logits=p.presence_logits - 100)
    assert torch.equal(model(p, t).features, model(changed, t).features)


def test_explicit_attached_option_allows_source_gradients():
    p, t = inputs()
    model = MembershipConditionedEventReadout(detach_sources=False).double()
    model(p, t).event_logits.square().sum().backward()
    assert p.membership_logits.grad.abs().sum() > 0 and p.centers_m.grad.abs().sum() > 0


@pytest.mark.parametrize("supported", [0, 2, 4])
def test_partial_or_zero_support_and_extreme_probabilities_finite(supported):
    p, t = inputs(); valid = torch.arange(4)[None] < supported
    tokens = replace(t, valid=valid)
    p = replace(p, query_supported=valid, member_supported=valid[:, :, None] & valid[:, None, :],
                membership_logits=torch.full_like(p.membership_logits, -10000))
    out = MembershipConditionedEventReadout().double()(p, tokens)
    assert torch.isfinite(out.features).all() and torch.isfinite(out.event_logits).all()
    assert torch.count_nonzero(out.event_logits[~valid]) == 0
    assert torch.count_nonzero(out.features[~valid]) == 0
    assert torch.count_nonzero(out.features[..., 8:17]) == 0


def test_all_64_queries_retained_no_member_selection():
    positions = torch.arange(192, dtype=torch.float64).reshape(1, 64, 3)
    tangent = torch.zeros_like(positions); tangent[..., 0] = 1
    tokens = AxisTokens(positions, tangent, torch.ones(1, 64, dtype=torch.bool))
    out = MembershipConditionedEventReadout().double()(prediction(positions.tolist()), tokens)
    assert out.event_logits.shape == (1, 64, 3) and out.query_supported.sum() == 64


@pytest.mark.parametrize("field,value", [("membership_logits", float("nan")), ("centers_m", float("inf")),
                                       ("uncertainty", 2.)])
def test_invalid_values_fail(field, value):
    p, t = inputs()
    p = replace(p, **{field: torch.full_like(getattr(p, field), value)})
    with pytest.raises(ValueError): membership_event_features(p, t)


@pytest.mark.parametrize("fault", ["shape", "dtype", "query_mask", "member_mask", "head_dtype", "switch"])
def test_contract_drift_fails(fault):
    p, t = inputs(); model = MembershipConditionedEventReadout().double()
    if fault == "shape": p = replace(p, centers_m=p.centers_m[:, :2])
    if fault == "dtype": p = replace(p, membership_logits=p.membership_logits.float())
    if fault == "query_mask": p = replace(p, query_supported=torch.zeros_like(p.query_supported))
    if fault == "member_mask": p = replace(p, member_supported=torch.zeros_like(p.member_supported))
    if fault == "head_dtype": model = model.float()
    if fault == "switch": model.use_membership = 1
    with pytest.raises(ValueError): model(p, t)
