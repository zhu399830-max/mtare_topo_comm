"""Synthetic operator/training-loop checks, never experimental performance."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from mtare_topo.representation.gse_partial_structure_training import (
    BRANCHES, PartialStructureExample, PartialTrainingConfig,
    paired_schedule, train_partial_structure,
)
from mtare_topo.representation.gse_region_queries import RegionQueryHead, RegionTargets


def fixture():
    examples = []
    for row in range(3):
        axes = torch.tensor([[[[-2., 0., 0.], [-1., 0., 0.], [0., 0., 0.]],
                              [[0., 0., 0.], [0., 1., 0.], [0., 2., 0.]]]])
        axes[..., 2] += row / 10
        m = 1 + row % 2
        target = RegionTargets(
            torch.zeros(1, m, 3), torch.ones(1, m, dtype=torch.bool),
            torch.ones(1, m, dtype=torch.long), torch.ones(1, m, dtype=torch.bool),
            torch.tensor([[[0., 1., 1., 0.]]]).expand(1, m, 4).clone(),
            torch.ones(1, m, 4, dtype=torch.bool), torch.zeros(1, dtype=torch.bool))
        examples.append(PartialStructureExample(axes.requires_grad_(), target))
    return {name: deepcopy(examples) for name in BRANCHES}


def config(**updates):
    return replace(PartialTrainingConfig(seed=2, steps=3, batch_size=2, lr=.001, device="cpu"), **updates)


def test_three_paired_heads_finite_backward_schedule_initialization_and_no_input_grads():
    data = fixture()
    before = deepcopy(data)
    rng = torch.random.get_rng_state().clone()
    result = train_partial_structure(data, config(), hidden=4)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert set(result.heads) == set(BRANCHES)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2)
        expected = RegionQueryHead(hidden=4).state_dict()
    for key, value in expected.items():
        assert torch.equal(value, result.initial_state[key])
    for name in BRANCHES:
        rows = result.history[name]
        assert len(rows) == 3
        assert [r["sample_indices"] for r in rows] == result.schedule
        assert [r["optimizer_steps"] for r in rows] == [1, 2, 3]
        assert all(r["gradient_l2"] > 0 and r["gradient_tensor_count"] > 0 for r in rows)
        assert all(torch.isfinite(torch.tensor(list(r["loss"].values()))).all() for r in rows)
        assert all(r["counts"]["presence_negative"] == 0 for r in rows)
        assert any(not torch.equal(value, result.initial_state[key])
                   for key, value in result.heads[name].state_dict().items())
        for original, current in zip(before[name], data[name]):
            assert current.axes.grad is None
            assert torch.equal(original.axes, current.axes)
    assert result.heads["gt_axes"].use_relations
    assert not result.heads["predicted_no_relations"].use_relations
    # Equal GT/pred inputs AND independent equal targets imply equal histories.
    assert result.history["gt_axes"] == result.history["predicted_axes"]


def test_same_seed_repeats_exactly():
    a = train_partial_structure(fixture(), config(), hidden=4)
    b = train_partial_structure(fixture(), config(), hidden=4)
    assert a.schedule == b.schedule and a.history == b.history
    for name in BRANCHES:
        for key, value in a.heads[name].state_dict().items():
            assert torch.equal(value, b.heads[name].state_dict()[key])


def test_different_gt_geometry_and_loss_mapping_are_not_reused_for_predictions():
    data = fixture()
    for row, example in enumerate(data["gt_axes"]):
        axes = example.axes.detach() + 8
        t = replace(example.target, centers_m=example.target.centers_m + 8)
        data["gt_axes"][row] = PartialStructureExample(axes, t)
    result = train_partial_structure(data, config(steps=1), hidden=4)
    assert result.history["gt_axes"][0]["loss"] != result.history["predicted_axes"][0]["loss"]
    assert result.history["gt_axes"][0]["sample_indices"] == result.history["predicted_axes"][0]["sample_indices"]


def test_schedule_is_without_replacement_each_epoch_and_final_short_batch():
    schedule = paired_schedule(5, config(steps=6, batch_size=2))
    assert [len(x) for x in schedule] == [2, 2, 1, 2, 2, 1]
    assert sorted(sum(schedule[:3], [])) == list(range(5))
    assert sorted(sum(schedule[3:], [])) == list(range(5))
    assert schedule == paired_schedule(5, config(steps=6, batch_size=2))


@pytest.mark.parametrize("updates", [dict(seed=True), dict(seed=-1), dict(steps=0),
    dict(batch_size=0), dict(lr=0), dict(lr=float("nan")), dict(lr=True), dict(device="meta")])
def test_invalid_explicit_config(updates):
    with pytest.raises(ValueError):
        train_partial_structure(fixture(), config(**updates), hidden=4)


def test_complete_labels_refused():
    data = fixture()
    data["gt_axes"][0].target.label_complete[:] = True
    with pytest.raises(ValueError, match="label_complete"):
        train_partial_structure(data, config(), hidden=4)


@pytest.mark.parametrize("field", ["axes", "members"])
def test_direct_ablation_drift_refused(field):
    data = fixture()
    with torch.no_grad():
        if field == "axes":
            data["predicted_no_relations"][0].axes[0, 0, 0, 0] += 1
        else:
            data["predicted_no_relations"][0].target.members[0, 0, 0] = 1
    with pytest.raises(ValueError, match="drift"):
        train_partial_structure(data, config(), hidden=4)


def test_no_supervision_is_not_zero_loss_success():
    data = fixture()
    for examples in data.values():
        for e in examples:
            e.target.center_valid[:] = False
            e.target.event_valid[:] = False
            e.target.member_valid[:] = False
    with pytest.raises(ValueError, match="no usable supervision"):
        train_partial_structure(data, config(), hidden=4)


def test_unknown_values_nan_are_allowed_without_changing_ablation_targets():
    data = fixture()
    for examples in data.values():
        for e in examples:
            e.target.member_valid[:, :, 0] = False
            e.target.members[:, :, 0] = float("nan")
    result = train_partial_structure(data, config(steps=1), hidden=4)
    assert all(torch.isfinite(torch.tensor(result.history[name][0]["loss"]["total"])) for name in BRANCHES)


def test_nonfinite_known_target_and_axis_refused():
    data = fixture()
    data["gt_axes"][0].target.centers_m[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="known target"):
        train_partial_structure(data, config(), hidden=4)
    data = fixture()
    with torch.no_grad():
        data["gt_axes"][0].axes[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        train_partial_structure(data, config(), hidden=4)


def test_partial_masks_are_not_silently_cast():
    data = fixture()
    e = data["gt_axes"][0]
    data["gt_axes"][0] = replace(e, target=replace(e.target, member_valid=e.target.member_valid.int()))
    with pytest.raises(ValueError, match="original target"):
        train_partial_structure(data, config(), hidden=4)


def test_actual_32_axes_64_queries_and_declared_head_width():
    data = fixture()
    for name in BRANCHES:
        for i, e in enumerate(data[name]):
            data[name][i] = replace(e, axes=e.axes.repeat(1, 16, 1, 1),
                target=replace(e.target, members=e.target.members.repeat(1, 1, 16),
                               member_valid=e.target.member_valid.repeat(1, 1, 16)))
    result = train_partial_structure(data, config(steps=1), hidden=64)
    assert sum(p.numel() for p in result.heads["gt_axes"].parameters()) == 26826
    assert all(r[0]["counts"]["membership"] >= 64 for r in result.history.values())


def test_every_optimizer_starts_from_identical_parameters_and_only_updates_head(monkeypatch):
    original = torch.optim.Adam
    starts = []
    def capturing_adam(parameters, **kwargs):
        parameters = list(parameters)
        starts.append([p.detach().clone() for p in parameters])
        return original(parameters, **kwargs)
    monkeypatch.setattr(torch.optim, "Adam", capturing_adam)
    train_partial_structure(fixture(), config(steps=1), hidden=4)
    assert len(starts) == 3
    for index in (1, 2):
        assert len(starts[index]) == len(starts[0])
        assert all(torch.equal(a, b) for a, b in zip(starts[0], starts[index]))


def test_nonfinite_optimizer_update_is_detected(monkeypatch):
    original = torch.optim.Adam.step
    def bad_step(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        with torch.no_grad():
            self.param_groups[0]["params"][0].flatten()[0] = float("nan")
        return result
    monkeypatch.setattr(torch.optim.Adam, "step", bad_step)
    with pytest.raises(FloatingPointError, match="updated head"):
        train_partial_structure(fixture(), config(steps=1), hidden=4)
