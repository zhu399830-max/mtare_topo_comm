"""Synthetic CPU parity/population checks, not experimental accuracy claims."""
from copy import deepcopy

import pytest
import torch

import mtare_topo.representation.gse_class_balance_training as balanced
from mtare_topo.representation.gse_geometry_bound_training import train_geometry_bound_structure
from tests.v3.unit.test_gse_geometry_bound_training import inputs
from tests.v3.unit.test_gse_partial_structure_training import config


def data():
    values = inputs()
    for examples in values.values():
        examples[0].target.events[0, 0] = 0
        examples[0].target.members[0, 0, 1] = 0
    return values  # four target rows: members (9 negative, 7 positive), events (1, 3, 0)


def states_equal(a, b):
    assert a.schedule == b.schedule
    for key, value in a.initial_state.items():
        assert torch.equal(value, b.initial_state[key])
    for name in balanced.BRANCHES:
        for key, value in a.heads[name].state_dict().items():
            assert torch.equal(value, b.heads[name].state_dict()[key])


def test_00_exact_geometry_training_history_states_schedule_and_rng():
    before_rng = torch.random.get_rng_state().clone()
    old = train_geometry_bound_structure(data(), config(), hidden=4)
    new = balanced.train_class_balanced_structure(data(), config(), hidden=4)
    assert old.history == new.history
    states_equal(old, new)
    assert torch.equal(before_rng, torch.random.get_rng_state())


@pytest.mark.parametrize("member,event", [((9, 7), None), (None, (1, 3, 0)), ((9, 7), (1, 3, 0))])
def test_factors_real_finite_updates_repeat_same_init_schedule_and_immutable_inputs(member, event):
    values = data()
    before = deepcopy(values)
    kwargs = dict(member_class_counts=member, event_class_counts=event)
    first = balanced.train_class_balanced_structure(values, config(), hidden=4, **kwargs)
    second = balanced.train_class_balanced_structure(data(), config(), hidden=4, **kwargs)
    original = train_geometry_bound_structure(data(), config(), hidden=4)
    assert first.history == second.history
    states_equal(first, second)
    assert first.schedule == original.schedule
    assert all(torch.equal(v, original.initial_state[k]) for k, v in first.initial_state.items())
    for name in balanced.BRANCHES:
        assert first.heads[name].use_relations == (name != "predicted_no_relations")
        for record in first.history[name]:
            assert record["gradient_l2"] > 0
            assert record["geometry_counts"] and record["supervision_counts"] and record["denominators"]
            assert torch.isfinite(torch.tensor(list(record["loss"].values()))).all()
            factors = record["class_balance"]
            assert set(factors) == ({"membership"} if member else set()) | ({"event"} if event else set())
            for task, counts in (("membership", member), ("event", event)):
                if counts:
                    assert factors[task]["global_class_counts"] == list(counts)
                    weights = [sum(counts) / (sum(n > 0 for n in counts) * n) if n else 0 for n in counts]
                    assert factors[task]["weights"] == torch.tensor(weights).tolist()
        for old, current in zip(before[name], values[name]):
            assert torch.equal(old.axes, current.axes) and current.axes.grad is None
            assert torch.equal(old.target.members, current.target.members)


def test_stub_unweighted_loss_proves_enabled_loop_only_changes_loss_and_configuration(monkeypatch):
    from mtare_topo.representation.gse_geometry_bound_losses import geometry_bound_region_losses
    calls = []
    def stub(prediction, target, **kwargs):
        calls.append(kwargs)
        return geometry_bound_region_losses(prediction, target)
    monkeypatch.setattr(balanced, "class_balanced_region_losses", stub)
    old = train_geometry_bound_structure(data(), config(), hidden=4)
    new = balanced.train_class_balanced_structure(data(), config(), hidden=4,
        member_class_counts=(9, 7), event_class_counts=(1, 3, 0))
    states_equal(old, new)
    for name in balanced.BRANCHES:
        assert old.history[name] == [{k: v for k, v in r.items() if k != "class_balance"}
                                     for r in new.history[name]]
    assert calls == [dict(member_class_counts=(9, 7), event_class_counts=(1, 3, 0))] * 9


@pytest.mark.parametrize("factor", ["membership", "event"])
def test_later_branch_population_mismatch_stops_before_any_head_optimizer_or_callback(monkeypatch, factor):
    values = data()
    for name in ("predicted_axes", "predicted_no_relations"):
        if factor == "membership":
            values[name][0].target.member_valid[0, 0, 0] = False
        else:
            values[name][0].target.event_valid[0, 0] = False
    def forbidden(*args, **kwargs): raise AssertionError("must validate every population first")
    monkeypatch.setattr(balanced, "RegionQueryHead", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    with pytest.raises(ValueError, match=f"predicted_axes: full-population {factor} counts"):
        balanced.train_class_balanced_structure(values, config(), hidden=4,
            member_class_counts=(9, 7), event_class_counts=(1, 3, 0), on_step=forbidden)


def test_numerically_unsupported_and_unknown_labels_not_in_global_counts():
    values = data()
    for examples in values.values():
        for example in examples:
            with torch.no_grad(): example.axes[:, 0, 0].copy_(example.axes[:, 0, 1])
        # Token 0 is unsupported for all four target rows: four negative labels.
        examples[0].target.member_valid[0, 0, 1] = False
        examples[0].target.members[0, 0, 1] = float("nan")
        examples[1].target.event_valid[0, 0] = False
        examples[1].target.events[0, 0] = -1
    result = balanced.train_class_balanced_structure(values, config(steps=1), hidden=4,
        member_class_counts=(4, 7), event_class_counts=(1, 2, 0))
    assert all(result.history[name][0]["class_balance"]["membership"]["global_class_counts"] == [4, 7]
               for name in balanced.BRANCHES)


@pytest.mark.parametrize("kwargs", [dict(member_class_counts=(True, 7)), dict(member_class_counts=[9, 7]),
    dict(member_class_counts=(9., 7)), dict(member_class_counts=(-1, 7)),
    dict(member_class_counts=(0, 16)), dict(event_class_counts=(0, 4, 0)),
    dict(event_class_counts=(1, 3)), dict(event_class_counts=(2, 2, 0))])
def test_invalid_or_wrong_global_counts_rejected_before_optimizer(monkeypatch, kwargs):
    def forbidden(*args, **kw): raise AssertionError("invalid population must not create optimizer")
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    with pytest.raises(ValueError):
        balanced.train_class_balanced_structure(data(), config(), hidden=4, **kwargs)


def test_callback_copy_includes_weights_and_stops_without_retry():
    calls = []
    def callback(name, record):
        calls.append((name, record["step"]))
        record["class_balance"]["membership"]["weights"][0] = -100
    result = balanced.train_class_balanced_structure(data(), config(steps=1), hidden=4,
        member_class_counts=(9, 7), on_step=callback)
    assert calls == [(name, 1) for name in balanced.BRANCHES]
    assert all(result.history[name][0]["class_balance"]["membership"]["weights"][0] > 0
               for name in balanced.BRANCHES)
    calls.clear()
    def stop(name, record):
        calls.append((name, record["step"]))
        raise RuntimeError("stop")
    with pytest.raises(RuntimeError, match="stop"):
        balanced.train_class_balanced_structure(data(), config(), hidden=4,
            member_class_counts=(9, 7), on_step=stop)
    assert calls == [("gt_axes", 1)]


def test_missing_supervision_stops_before_update(monkeypatch):
    original = balanced.class_balanced_region_losses
    def empty(*args, **kwargs):
        loss = original(*args, **kwargs)
        loss["has_supervision"] = False
        return loss
    monkeypatch.setattr(balanced, "class_balanced_region_losses", empty)
    def forbidden(*args, **kwargs): raise AssertionError("no supervision cannot update")
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    with pytest.raises(ValueError, match="no usable supervision"):
        balanced.train_class_balanced_structure(data(), config(), hidden=4, member_class_counts=(9, 7))
