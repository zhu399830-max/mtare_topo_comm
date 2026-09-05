from dataclasses import replace
import inspect

import pytest
import torch

from mtare_topo.representation.gse_region_queries import (
    AxisTokens, RegionQueryHead, RegionTargets, region_set_losses,
    relation_features, tokens_from_axes,
)


def fixture():
    positions = torch.tensor([[[2., 0., 0.], [-2., 0., 0.], [0., 2., 0.],
                               [12., 0., 3.], [8., 0., 3.], [10., 2., 3.]]])
    tangents = torch.tensor([[[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]]]).repeat(1, 2, 1)
    return AxisTokens(positions, tangents, torch.ones(1, 6, dtype=torch.bool))


def targets(*, complete=False):
    return RegionTargets(torch.tensor([[[0., 0., 0.], [10., 0., 3.]]]), torch.ones(1, 2, dtype=torch.bool),
                         torch.tensor([[1, 1]]), torch.ones(1, 2, dtype=torch.bool),
                         torch.tensor([[[1., 1., 1., 0., 0., 0.], [0., 0., 0., 1., 1., 1.]]]),
                         torch.ones(1, 2, 6, dtype=torch.bool), torch.tensor([complete]))


def test_all_predicted_candidates_are_kept_without_teacher_arguments():
    axes = torch.randn(2, 32, 3, 3)
    value = tokens_from_axes(axes)
    prediction = RegionQueryHead()(value)
    assert prediction.centers_m.shape == (2, 64, 3)
    assert prediction.membership_logits.shape == (2, 64, 64)
    assert set(inspect.signature(tokens_from_axes).parameters) == {"axes"}
    assert set(inspect.signature(RegionQueryHead.forward).parameters) == {"self", "value"}
    assert set(value.__dict__) == {"positions_m", "tangents", "valid"}


def test_two_regions_get_two_matches_not_one_global_event():
    pred = RegionQueryHead()(fixture())
    result = region_set_losses(pred, targets())
    assert pred.event_logits.shape == (1, 6, 3)
    assert result["counts"]["matched"] == 2
    assert len({query for _, query, _ in result["matches"]}) == 2
    assert result["counts"]["membership"] == 12


def test_common_yaw_rotates_centers_and_preserves_structure():
    value = fixture()
    a = torch.tensor(.73); c, s = a.cos(), a.sin()
    r = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    rotated = replace(value, positions_m=value.positions_m @ r.T, tangents=value.tangents @ r.T)
    model = RegionQueryHead().eval()
    first, second = model(value), model(rotated)
    torch.testing.assert_close(first.centers_m @ r.T, second.centers_m, atol=3e-6, rtol=1e-5)
    for field in ("event_logits", "presence_logits", "membership_logits", "uncertainty"):
        torch.testing.assert_close(getattr(first, field), getattr(second, field), atol=1e-6, rtol=1e-5)


def test_permutations_reorder_both_query_and_member_axes():
    value, order = fixture(), torch.tensor([5, 2, 0, 4, 1, 3])
    permuted = AxisTokens(**{k: v[:, order] for k, v in value.__dict__.items()})
    model = RegionQueryHead().eval()
    a, b = model(value), model(permuted)
    torch.testing.assert_close(a.centers_m[:, order], b.centers_m)
    torch.testing.assert_close(a.event_logits[:, order], b.event_logits)
    torch.testing.assert_close(a.membership_logits[:, order][:, :, order], b.membership_logits)
    label = targets()
    la = region_set_losses(a, label)
    lb = region_set_losses(b, replace(label, members=label.members[:, :, order], member_valid=label.member_valid[:, :, order]))
    torch.testing.assert_close(la["total"], lb["total"])


def test_target_order_has_no_effect_on_loss():
    pred, target = RegionQueryHead()(fixture()), targets()
    shuffled = RegionTargets(**{k: (v if k == "label_complete" else v.flip(1)) for k, v in target.__dict__.items()})
    torch.testing.assert_close(region_set_losses(pred, target)["total"], region_set_losses(pred, shuffled)["total"])


def test_angular_layout_survives_unary_equivalence_and_ablation_removes_it():
    value = fixture()
    value = AxisTokens(**{k: v[:, :3] for k, v in value.__dict__.items()})
    p = value.positions_m.clone(); p[0, 2] = torch.tensor([1., 3.**.5, 0.])
    t = value.tangents.clone(); t[0, 2] = p[0, 2] / 2
    changed = replace(value, positions_m=p, tangents=t)
    f1 = relation_features(value.positions_m, value.tangents, value)[0]
    f2 = relation_features(p, t, changed)[0]
    assert not torch.allclose(f1, f2)
    torch.manual_seed(7)
    full = RegionQueryHead()
    no_combo = RegionQueryHead(use_relations=False); no_combo.load_state_dict(full.state_dict())
    for x, y in zip(full.parameters(), no_combo.parameters()): assert torch.equal(x, y)
    assert not torch.allclose(full(value).membership_logits, full(changed).membership_logits)
    torch.testing.assert_close(no_combo(value).membership_logits, no_combo(changed).membership_logits)
    torch.testing.assert_close(no_combo(value).event_logits, no_combo(changed).event_logits)


def test_stacked_relation_retains_signed_height():
    v = fixture(); p = v.positions_m.clone(); p[:, 3:, 2] += 4
    first, _ = relation_features(v.positions_m, v.tangents, v)
    second, _ = relation_features(p, v.tangents, replace(v, positions_m=p))
    assert second[0, 0, 3, 1] - first[0, 0, 3, 1] == pytest.approx(4/50)


def test_identical_geometry_does_not_generate_identity_features():
    v = fixture(); v = AxisTokens(v.positions_m[:, :1].repeat(1, 2, 1), v.tangents[:, :1].repeat(1, 2, 1), v.valid[:, :2])
    pred = RegionQueryHead()(v)
    # Shared GEMM rows can differ by float32 rounding; no slot embedding or
    # index is supplied. Use a numeric tolerance, not a learned identity gate.
    torch.testing.assert_close(pred.centers_m[:, 0], pred.centers_m[:, 1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(pred.membership_logits[:, 0], pred.membership_logits[:, 1], atol=1e-6, rtol=1e-6)


def test_padding_cannot_change_valid_outputs():
    value = fixture(); value = replace(value, valid=torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool))
    p = value.positions_m.clone(); p[:, 3:] = 1e4
    t = value.tangents.clone(); t[:, 3:] = -1e4
    model = RegionQueryHead()
    a, b = model(value), model(replace(value, positions_m=p, tangents=t))
    for field in a.__dict__: torch.testing.assert_close(getattr(a, field), getattr(b, field))


def test_all_unsupported_is_unknown_not_terminal():
    value = fixture(); value = replace(value, valid=torch.zeros_like(value.valid))
    pred = RegionQueryHead()(value)
    assert not pred.query_supported.any()
    assert torch.equal(pred.uncertainty, torch.ones(1, 6))
    assert torch.equal(pred.centers_m, torch.zeros(1, 6, 3))
    result = region_set_losses(pred, targets())
    assert result["counts"]["unmatched_targets"] == 2
    assert not result["has_supervision"]
    assert result["total"] == 0


def test_unknown_labels_do_not_create_background_negatives():
    model, label = RegionQueryHead(), targets()
    pred = model(fixture())
    unknown = replace(label, centers_m=torch.full_like(label.centers_m, float("nan")),
                      center_valid=torch.zeros_like(label.center_valid), events=torch.full_like(label.events, -999),
                      event_valid=torch.zeros_like(label.event_valid), members=torch.full_like(label.members, float("nan")),
                      member_valid=torch.zeros_like(label.member_valid))
    result = region_set_losses(pred, unknown)
    assert result["total"] == 0 and not result["has_supervision"]
    assert result["counts"]["presence_negative"] == 0
    result["total"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_only_label_complete_permits_unmatched_presence_negatives():
    pred = RegionQueryHead()(fixture())
    incomplete, complete = region_set_losses(pred, targets()), region_set_losses(pred, targets(complete=True))
    assert incomplete["counts"]["presence_negative"] == 0
    assert complete["counts"]["presence_negative"] == 4
    assert complete["counts"]["presence_positive"] == 2


def test_every_head_branch_has_finite_gradient():
    model = RegionQueryHead(); pred = model(fixture())
    result = region_set_losses(pred, targets(complete=True)); result["total"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_no_combo_gradients_and_parameter_budget():
    model = RegionQueryHead(use_relations=False)
    result = region_set_losses(model(fixture()), targets()); result["total"].backward()
    assert sum(p.numel() for p in model.parameters()) < 100_000
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_axis_reversal_just_swaps_end_tokens():
    axis = torch.tensor([[[[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]]]])
    a, b = tokens_from_axes(axis), tokens_from_axes(axis.flip(2))
    for name in a.__dict__: torch.testing.assert_close(getattr(a, name).flip(1), getattr(b, name))


def test_degenerate_axes_kept_but_not_given_fake_tangents():
    a = tokens_from_axes(torch.zeros(1, 32, 3, 3))
    assert a.valid.shape == (1, 64) and not a.valid.any()
    assert torch.isfinite(a.tangents).all()


@pytest.mark.parametrize("axis", [torch.zeros(1, 33, 3, 3), torch.zeros(1, 2, 3),
    torch.zeros(0, 1, 3, 3), torch.zeros(1, 1, 3, 3, dtype=torch.long), torch.full((1, 1, 3, 3), float("nan"))])
def test_invalid_axes_refused(axis):
    with pytest.raises(ValueError): tokens_from_axes(axis)


@pytest.mark.parametrize("field,change", [
    ("valid", lambda v: v.float()), ("positions_m", lambda v: v.double()),
    ("positions_m", lambda v: v * float("nan")), ("tangents", lambda v: v * 2),
    ("tangents", lambda v: v[:, :, :2]),
])
def test_invalid_tokens_refused(field, change):
    value = fixture()
    with pytest.raises(ValueError): RegionQueryHead()(replace(value, **{field: change(getattr(value, field))}))


@pytest.mark.parametrize("field,change", [
    ("events", lambda v: v.float()), ("events", lambda v: v * 10),
    ("members", lambda v: v + 2), ("center_valid", lambda v: v.long()),
    ("centers_m", lambda v: v * float("nan")), ("members", lambda v: v.double()),
])
def test_invalid_targets_refused(field, change):
    label = targets()
    with pytest.raises(ValueError):
        region_set_losses(RegionQueryHead()(fixture()), replace(label, **{field: change(getattr(label, field))}))


def test_empty_target_set_only_supervises_background_when_complete():
    pred, label = RegionQueryHead()(fixture()), targets(complete=True)
    empty = RegionTargets(**{k: (v if k == "label_complete" else v[:, :0]) for k, v in label.__dict__.items()})
    result = region_set_losses(pred, empty)
    assert result["counts"]["matched"] == 0 and result["counts"]["presence_negative"] == 6


def test_prediction_repeat_is_bitwise_identical():
    model, value = RegionQueryHead().eval(), fixture()
    first, second = model(value), model(value)
    for name in first.__dict__: assert torch.equal(getattr(first, name), getattr(second, name))


def test_unusable_member_only_label_is_unmatched_not_positive_presence():
    value = fixture(); value = replace(value, valid=torch.tensor([[True, True, True, False, False, False]]))
    label = targets()
    label = replace(label, center_valid=torch.zeros_like(label.center_valid), event_valid=torch.zeros_like(label.event_valid),
                    member_valid=torch.tensor([[[False, False, False, True, True, True]]]).repeat(1, 2, 1))
    result = region_set_losses(RegionQueryHead()(value), label)
    assert result["counts"]["targets"] == result["counts"]["unmatched_targets"] == 2
    assert result["counts"]["unsupported_member_targets"] == 2
    assert result["counts"]["presence_positive"] == 0
    assert not result["has_supervision"] and result["total"] == 0


def test_nonfinite_prediction_cannot_return_successful_loss():
    pred = RegionQueryHead()(fixture())
    with pytest.raises(ValueError, match="nonfinite prediction"):
        region_set_losses(replace(pred, presence_logits=pred.presence_logits * float("nan")), targets())


@pytest.mark.parametrize("complete", [False, True])
def test_geometrically_tied_queries_match_presence_not_array_index(complete):
    value = fixture(); value = AxisTokens(**{k: v[:, :2] for k, v in value.__dict__.items()})
    pred = RegionQueryHead()(value)
    pred = replace(pred, centers_m=torch.zeros(1, 2, 3), presence_logits=torch.tensor([[-10., 10.]]))
    target = RegionTargets(torch.zeros(1, 1, 3), torch.tensor([[True]]), torch.tensor([[-999]]), torch.tensor([[False]]),
                           torch.zeros(1, 1, 2), torch.zeros(1, 1, 2, dtype=torch.bool), torch.tensor([complete]))
    a = region_set_losses(pred, target)
    permuted = replace(pred, presence_logits=pred.presence_logits.flip(1))
    b = region_set_losses(permuted, target)
    torch.testing.assert_close(a["total"], b["total"])
    assert a["matches"] == [(0, 1, 0)] and b["matches"] == [(0, 0, 0)]


def test_near_resolution_axis_support_does_not_flip_under_yaw45():
    axes = torch.tensor([[[[39.99988, 0., 0.], [40., 0., 0.], [40.00012, 0., 0.]]]])
    a = torch.tensor(torch.pi / 4); c, s = a.cos(), a.sin()
    rotation = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    first, second = tokens_from_axes(axes), tokens_from_axes(axes @ rotation.T)
    assert torch.equal(first.valid, second.valid) and not first.valid.any()


def test_real_axis_bridge_to_queries_common_yaw():
    axes = torch.tensor([[[[-3., 0., 0.], [-1., 0., 0.], [1., 0., 0.]],
                          [[0., 1., 2.], [0., 2., 3.], [0., 3., 4.]]]])
    a = torch.tensor(.73); c, s = a.cos(), a.sin()
    rotation = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    model = RegionQueryHead()
    first, second = model(tokens_from_axes(axes)), model(tokens_from_axes(axes @ rotation.T))
    torch.testing.assert_close(first.centers_m @ rotation.T, second.centers_m, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(first.membership_logits, second.membership_logits)


def test_common_yaw_preserves_set_loss_not_only_forward():
    value, label = fixture(), targets(complete=True)
    a = torch.tensor(.73); c, s = a.cos(), a.sin()
    rotation = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    model = RegionQueryHead()
    original = region_set_losses(model(value), label)
    rotated = region_set_losses(model(replace(value, positions_m=value.positions_m @ rotation.T,
        tangents=value.tangents @ rotation.T)), replace(label, centers_m=label.centers_m @ rotation.T))
    torch.testing.assert_close(original["total"], rotated["total"])


def test_more_targets_than_queries_exposes_missing_targets():
    v = fixture(); v = AxisTokens(**{k: x[:, :1] for k, x in v.__dict__.items()})
    label = targets(); label = replace(label, members=label.members[:, :, :1], member_valid=label.member_valid[:, :, :1])
    result = region_set_losses(RegionQueryHead()(v), label)
    assert result["counts"]["targets"] == 2
    assert result["counts"]["matched"] == 1 and result["counts"]["unmatched_targets"] == 1


def test_negative_only_membership_cannot_certify_region_presence():
    label = targets()
    label = replace(label, center_valid=torch.zeros_like(label.center_valid), event_valid=torch.zeros_like(label.event_valid),
                    members=torch.zeros_like(label.members))
    result = region_set_losses(RegionQueryHead()(fixture()), label)
    assert result["counts"]["unconfirmed_presence_targets"] == 2
    assert result["counts"]["presence_positive"] == 0 and not result["has_supervision"]
