from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from mtare_topo.representation.gse_typed_composers import (
    ActionSetComposerInput,
    ActionSetRelationComposer,
    MetricChangeComposer,
    MetricChangeComposerInput,
    action_set_composer_loss,
    metric_change_composer_loss,
    typed_composer_input_contract,
)


def _action_input(batch: int = 3) -> ActionSetComposerInput:
    generator = torch.Generator().manual_seed(17)
    angle = torch.rand(batch, 5, 6, generator=generator) * (2.0 * torch.pi) - torch.pi
    bearing = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    count = torch.softmax(torch.randn(batch, 5, 7, generator=generator), dim=-1)
    transport = torch.softmax(torch.randn(batch, 4, 6, 7, generator=generator), dim=-1)
    return ActionSetComposerInput(
        token_bearing_unit=bearing,
        token_existence_probability=torch.sigmoid(torch.randn(batch, 5, 6, generator=generator)),
        token_opening_width_m=torch.rand(batch, 5, 6, generator=generator) * 4.0 + 0.5,
        token_vertical_profile_m=torch.randn(batch, 5, 6, 4, generator=generator),
        token_geometry_uncertainty=torch.rand(batch, 5, 6, 5, generator=generator) + 0.05,
        token_count_probability=count,
        transport_row_probability=transport,
        transport_reveal_probability=torch.sigmoid(torch.randn(batch, 4, 6, generator=generator)),
        valid_history_mask=torch.ones(batch, 5, dtype=torch.bool),
    )


def _metric_input(batch: int = 3) -> MetricChangeComposerInput:
    generator = torch.Generator().manual_seed(29)
    geometry = torch.randn(batch, 5, 4, generator=generator)
    geometry[..., 0] = geometry[..., 0].abs() + 2.0
    geometry[..., 1] = geometry[..., 1].abs() + 1.5
    return MetricChangeComposerInput(
        geometry_sequence=geometry,
        geometry_uncertainty=torch.rand(batch, 5, 4, generator=generator) + 0.01,
        valid_history_mask=torch.ones(batch, 5, dtype=torch.bool),
    )


def _permute_action_tokens(
    inputs: ActionSetComposerInput, permutation: torch.Tensor,
) -> ActionSetComposerInput:
    row = inputs.transport_row_probability[:, :, permutation]
    match = row[..., :6][..., permutation]
    row = torch.cat((match, row[..., 6:]), dim=-1)
    return replace(
        inputs,
        token_bearing_unit=inputs.token_bearing_unit[:, :, permutation],
        token_existence_probability=inputs.token_existence_probability[:, :, permutation],
        token_opening_width_m=inputs.token_opening_width_m[:, :, permutation],
        token_vertical_profile_m=inputs.token_vertical_profile_m[:, :, permutation],
        token_geometry_uncertainty=inputs.token_geometry_uncertainty[:, :, permutation],
        transport_row_probability=row,
        transport_reveal_probability=inputs.transport_reveal_probability[:, :, permutation],
    )


def test_typed_contract_has_only_explicit_deployment_geometry() -> None:
    contract = typed_composer_input_contract()
    assert contract["action_fields"] == tuple(field.name for field in fields(ActionSetComposerInput))
    assert contract["metric_fields"] == tuple(field.name for field in fields(MetricChangeComposerInput))
    exposed = set(contract["action_fields"]) | set(contract["metric_fields"])
    assert exposed.isdisjoint(contract["forbidden_inputs"])
    assert contract["history_frames"] == 5
    assert contract["backprojection_support_steps_ago"] == (4, 3, 2, 1, 0)


def test_action_composer_is_token_permutation_and_rotation_invariant() -> None:
    torch.manual_seed(3)
    model = ActionSetRelationComposer().eval()
    inputs = _action_input()
    expected = model(inputs)
    permutation = torch.tensor((3, 0, 5, 2, 1, 4))
    permuted = model(_permute_action_tokens(inputs, permutation))
    torch.testing.assert_close(permuted.event_probability, expected.event_probability, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(permuted.refusal_probability, expected.refusal_probability, atol=1e-6, rtol=1e-6)

    rotation = torch.tensor(0.73)
    sine, cosine = torch.sin(rotation), torch.cos(rotation)
    bearing = inputs.token_bearing_unit
    rotated = torch.stack((
        bearing[..., 0] * cosine + bearing[..., 1] * sine,
        bearing[..., 1] * cosine - bearing[..., 0] * sine,
    ), dim=-1)
    actual = model(replace(inputs, token_bearing_unit=rotated))
    torch.testing.assert_close(actual.event_probability, expected.event_probability, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.refusal_probability, expected.refusal_probability, atol=1e-6, rtol=1e-6)


def test_both_composers_are_batch_permutation_consistent_and_deterministic() -> None:
    torch.manual_seed(5)
    action_model = ActionSetRelationComposer().eval()
    metric_model = MetricChangeComposer().eval()
    action = _action_input()
    metric = _metric_input()
    permutation = torch.tensor((2, 0, 1))

    action_expected = action_model(action).event_probability
    action_actual = action_model(ActionSetComposerInput(**{
        field.name: getattr(action, field.name)[permutation] for field in fields(action)
    })).event_probability
    torch.testing.assert_close(action_actual, action_expected[permutation], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(action_model(action).event_probability, action_expected, atol=0.0, rtol=0.0)

    metric_expected = metric_model(metric).event_probability
    metric_actual = metric_model(MetricChangeComposerInput(**{
        field.name: getattr(metric, field.name)[permutation] for field in fields(metric)
    })).event_probability
    torch.testing.assert_close(metric_actual, metric_expected[permutation], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(metric_model(metric).event_probability, metric_expected, atol=0.0, rtol=0.0)


def test_left_padding_is_ignored_and_backprojection_never_uses_invalid_history() -> None:
    torch.manual_seed(7)
    action_model = ActionSetRelationComposer().eval()
    metric_model = MetricChangeComposer().eval()
    action = _action_input(batch=1)
    metric = _metric_input(batch=1)
    mask = torch.tensor(((False, False, True, True, True),))

    action_masked = replace(action, valid_history_mask=mask)
    action_changed = replace(
        action_masked,
        token_opening_width_m=action.token_opening_width_m.clone(),
    )
    action_changed.token_opening_width_m[:, :2].fill_(999.0)
    torch.testing.assert_close(
        action_model(action_masked).event_probability,
        action_model(action_changed).event_probability,
        atol=0.0, rtol=0.0,
    )

    metric_masked = replace(metric, valid_history_mask=mask)
    changed_geometry = metric.geometry_sequence.clone()
    changed_geometry[:, :2] = torch.tensor((999.0, 999.0, 999.0, 999.0))
    metric_changed = replace(metric_masked, geometry_sequence=changed_geometry)
    expected = metric_model(metric_masked)
    actual = metric_model(metric_changed)
    torch.testing.assert_close(actual.event_probability, expected.event_probability, atol=0.0, rtol=0.0)
    assert torch.equal(expected.backprojection_probability[:, :2], torch.zeros(1, 2))
    assert bool(((expected.expected_steps_ago >= 0.0) & (expected.expected_steps_ago <= 2.0)).all())


def test_complete_losses_give_finite_gradients_to_every_parameter() -> None:
    torch.manual_seed(11)
    action_model = ActionSetRelationComposer()
    action_output = action_model(_action_input())
    action_loss = action_set_composer_loss(
        action_output, torch.tensor((0, 1, 2)), torch.tensor((1, 1, 0)),
    )["total"]
    action_loss.backward()
    assert torch.isfinite(action_loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in action_model.parameters())

    metric_model = MetricChangeComposer()
    metric_output = metric_model(_metric_input())
    metric_loss = metric_change_composer_loss(
        metric_output,
        torch.tensor((0, 1, 2)),
        torch.tensor((1, 1, 0)),
        torch.tensor((4, 3, 1)),
    )["total"]
    metric_loss.backward()
    assert torch.isfinite(metric_loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in metric_model.parameters())


def test_invalid_masks_and_probability_contracts_are_rejected() -> None:
    action = _action_input(batch=1)
    bad_mask = torch.tensor(((True, False, True, True, True),))
    with pytest.raises(ValueError, match="causal suffix"):
        ActionSetRelationComposer()(replace(action, valid_history_mask=bad_mask))
    bad_count = action.token_count_probability.clone()
    bad_count[..., 0] += 0.5
    with pytest.raises(ValueError, match="sum to one"):
        ActionSetRelationComposer()(replace(action, token_count_probability=bad_count))

    metric = _metric_input(batch=1)
    with pytest.raises(ValueError, match="causal suffix"):
        MetricChangeComposer()(replace(metric, valid_history_mask=bad_mask))

