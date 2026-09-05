"""Synthetic paired-loop parity and safety, not research accuracy tests."""
from copy import deepcopy

import pytest
import torch

import mtare_topo.representation.gse_geometry_bound_training as bound
from mtare_topo.representation.gse_partial_structure_training import train_partial_structure
from mtare_topo.representation.gse_region_queries import region_set_losses
from tests.v3.unit.test_gse_partial_structure_training import fixture, config


def inputs():
    values = fixture()
    # Distinct target centers avoid deliberate duplicate-center ambiguity in
    # the legacy fixture; target population and paired branches stay fixed.
    for examples in values.values():
        for example in examples:
            # Legacy axes meet at one exact shared endpoint: without relative
            # features those two proposals coincide, correctly causing UNKNOWN.
            # Use separated endpoints for the ordinary-loop success fixture.
            with torch.no_grad():
                example.axes[:, 1, :, 0].add_(3.)
            if example.target.centers_m.shape[1] == 2:
                example.target.centers_m[0, 1, 0] = 3
                example.target.centers_m[0, 1, 1] = 2
    return values


def test_real_geometry_loss_repeat_same_initial_schedule_finite_updates_and_no_input_gradients():
    data = inputs()
    before = deepcopy(data)
    rng = torch.random.get_rng_state().clone()
    first = bound.train_geometry_bound_structure(data, config(), hidden=4)
    assert torch.equal(rng, torch.random.get_rng_state())
    second = bound.train_geometry_bound_structure(inputs(), config(), hidden=4)
    assert first.history == second.history and first.schedule == second.schedule
    old = train_partial_structure(inputs(), config(), hidden=4)
    assert first.schedule == old.schedule
    assert all(torch.equal(value, old.initial_state[key]) for key, value in first.initial_state.items())
    for branch in bound.BRANCHES:
        assert first.heads[branch].use_relations == (branch != "predicted_no_relations")
        assert all(torch.equal(value, second.heads[branch].state_dict()[key])
                   for key, value in first.heads[branch].state_dict().items())
        assert [r["sample_indices"] for r in first.history[branch]] == first.schedule
        assert len(first.history[branch]) == 3
        assert all(r["counts"]["presence_negative"] == 0 for r in first.history[branch])
        assert all(r["geometry_counts"] and r["supervision_counts"] and r["denominators"] for r in first.history[branch])
        for prior, after in zip(before[branch], data[branch]):
            assert torch.equal(prior.axes, after.axes)
            assert after.axes.grad is None


def test_substitute_old_loss_gives_exact_old_history_weights_and_optimizer_behavior(monkeypatch):
    monkeypatch.setattr(bound, "geometry_bound_region_losses", region_set_losses)
    old = train_partial_structure(inputs(), config(), hidden=4)
    new = bound.train_geometry_bound_structure(inputs(), config(), hidden=4)
    assert old.schedule == new.schedule
    for branch in bound.BRANCHES:
        for original, candidate in zip(old.history[branch], new.history[branch]):
            assert original == {key: value for key, value in candidate.items() if key in original}
            assert candidate["geometry_counts"] == candidate["supervision_counts"] == candidate["denominators"] == {}
        assert all(torch.equal(value, new.heads[branch].state_dict()[key])
                   for key, value in old.heads[branch].state_dict().items())


def test_callback_receives_new_counts_only_after_success_and_cannot_mutate_history():
    callbacks = []
    def callback(branch, record):
        callbacks.append((branch, record["step"], deepcopy(record["supervision_counts"])))
        record["counts"]["center"] = -999
        record["supervision_counts"]["known_centers"] = -999
        record["sample_indices"].clear()
    result = bound.train_geometry_bound_structure(inputs(), config(steps=1), hidden=4, on_step=callback)
    assert [branch for branch, _, _ in callbacks] == list(bound.BRANCHES)
    for branch in bound.BRANCHES:
        row = result.history[branch][0]
        assert row["counts"]["center"] >= 0
        assert row["supervision_counts"]["known_centers"] > 0
        assert row["sample_indices"]


def test_callback_failure_stops_without_next_step_or_retry():
    calls = []
    def callback(branch, record):
        calls.append((branch, record["step"]))
        raise RuntimeError("synthetic callback stop")
    with pytest.raises(RuntimeError, match="synthetic callback stop"):
        bound.train_geometry_bound_structure(inputs(), config(), hidden=4, on_step=callback)
    assert calls == [("gt_axes", 1)]


def test_no_supervision_fails_before_optimizer_and_callback(monkeypatch):
    original = bound.geometry_bound_region_losses
    def no_supervision(prediction, target):
        value = original(prediction, target)
        value["has_supervision"] = False
        return value
    monkeypatch.setattr(bound, "geometry_bound_region_losses", no_supervision)
    def forbidden(*args, **kwargs): raise AssertionError("no update or callback allowed")
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    with pytest.raises(ValueError, match="no usable supervision"):
        bound.train_geometry_bound_structure(inputs(), config(), hidden=4, on_step=forbidden)


def test_actual_unknown_targets_fail():
    data = inputs()
    for examples in data.values():
        for example in examples:
            example.target.center_valid[:] = False
            example.target.event_valid[:] = False
            example.target.member_valid[:] = False
    with pytest.raises(ValueError, match="no usable supervision"):
        bound.train_geometry_bound_structure(data, config(), hidden=4)


def test_actual_all_ambiguous_geometry_fails_before_any_step(monkeypatch):
    data = inputs()
    for examples in data.values():
        for example in examples:
            with torch.no_grad(): example.axes[:, 1].copy_(example.axes[:, 0])
    def forbidden(*args, **kwargs): raise AssertionError("ambiguous geometry must not update")
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    with pytest.raises(ValueError, match="no usable supervision"):
        bound.train_geometry_bound_structure(data, config(), hidden=4, on_step=forbidden)


def test_unchanged_adam_learning_rate_and_paired_initial_weights(monkeypatch):
    original = torch.optim.Adam
    starts = []
    def capture(parameters, **kwargs):
        parameters = list(parameters)
        starts.append(([p.detach().clone() for p in parameters], kwargs))
        return original(parameters, **kwargs)
    monkeypatch.setattr(torch.optim, "Adam", capture)
    bound.train_geometry_bound_structure(inputs(), config(steps=1), hidden=4)
    assert len(starts) == 3
    for weights, kwargs in starts:
        assert kwargs == {"lr": config().lr}
        assert all(torch.equal(a, b) for a, b in zip(weights, starts[0][0]))


def test_nonfinite_loss_stops_before_step(monkeypatch):
    original = bound.geometry_bound_region_losses
    def nonfinite(prediction, target):
        loss = original(prediction, target)
        loss["total"] = loss["total"] * float("nan")
        return loss
    monkeypatch.setattr(bound, "geometry_bound_region_losses", nonfinite)
    with pytest.raises(FloatingPointError, match="nonfinite partial loss"):
        bound.train_geometry_bound_structure(inputs(), config(), hidden=4)


def test_callback_must_be_callable():
    with pytest.raises(ValueError, match="callable"):
        bound.train_geometry_bound_structure(inputs(), config(), hidden=4, on_step=3)


def test_direct_ablation_drift_still_refused():
    data = inputs()
    with torch.no_grad(): data["predicted_no_relations"][0].axes.add_(.1)
    with pytest.raises(ValueError, match="drift"):
        bound.train_geometry_bound_structure(data, config(), hidden=4)
