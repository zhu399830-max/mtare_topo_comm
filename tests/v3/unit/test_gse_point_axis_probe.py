"""Tiny synthetic paired-fit tests; no project data or checkpoint is loaded."""
import numpy as np
import pytest
import torch

from mtare_topo.representation.gse_point_axis_probe import (
    configure_variant, decide, evaluate, fit, sample_schedule, tensor_state_sha,
)
from mtare_topo.representation.gse_point_axis_readout import PointAxisReadout


def _fixture():
    torch.manual_seed(27)
    head = PointAxisReadout(model_dim=8, point_dim=4).double()
    generator = torch.Generator().manual_seed(16)
    points = torch.randn(1, 13, 3, generator=generator, dtype=torch.float64)
    valid = torch.ones(1, 13, dtype=torch.bool)
    memory = torch.randn(1, 3, 8, generator=generator, dtype=torch.float64)
    memory_xyz = torch.randn(1, 3, 3, generator=generator, dtype=torch.float64)
    sensor_indices = torch.arange(13)[None] % 3
    slots = torch.randn(1, 2, 8, generator=generator, dtype=torch.float64)
    student = (points, valid, memory, memory_xyz, sensor_indices, slots)
    target = torch.tensor([[[[-1., 2., 0.], [0., 2., 0.], [1., 2., 0.]],
                             [[-1., -2., 1.], [0., -2., 1.], [1., -2., 1.]]]], dtype=torch.float64)
    cache = [{"student": student, "target": target, "mask": torch.ones(1, 2, dtype=torch.bool),
              "task": "synthetic_only"}]
    return head, cache


def test_paired_initialization_is_exact_and_configuring_variants_does_not_mutate_initial():
    initial, cache = _fixture()
    before = tensor_state_sha(initial)
    raw, full = [configure_variant(initial, variant) for variant in ("raw_no_offset", "raw_slot_offset")]
    assert tensor_state_sha(raw) == tensor_state_sha(full) == before
    raw_axes, raw_metrics = evaluate(raw, cache)
    full_axes, full_metrics = evaluate(full, cache)
    assert np.array_equal(raw_axes, full_axes) and raw_metrics == full_metrics
    assert all(p.requires_grad for p in initial.parameters())
    assert not any(p.requires_grad for layer in (raw.offset, raw.slot_offset) for p in layer.parameters())
    assert all(p.requires_grad for p in full.parameters())
    assert tensor_state_sha(initial) == before


def test_no_offset_fit_keeps_offsets_zero_and_other_parameters_update():
    initial, cache = _fixture()
    raw = configure_variant(initial, "raw_no_offset")
    logs = []
    result = fit(raw, cache, [0, 0, 0], logs.append)
    assert result["steps"] == 3 and len(logs) == 3
    assert result["initial_state_sha256"] != result["final_state_sha256"]
    for layer in (raw.offset, raw.slot_offset):
        for parameter in layer.parameters():
            assert not parameter.any() and parameter.grad is None
    assert all(np.isfinite(row["loss_coordinate_l1_div50"]) for row in logs)
    assert [row["step"] for row in logs] == [1, 2, 3]


def test_full_fit_updates_candidate_offsets_with_finite_nonzero_gradients():
    initial, cache = _fixture()
    full = configure_variant(initial, "raw_slot_offset")
    result = fit(full, cache, [0, 0, 0], lambda row: None)
    assert result["initial_state_sha256"] != result["final_state_sha256"]
    for layer in (full.offset, full.slot_offset):
        assert any(parameter.abs().sum() > 0 for parameter in layer.parameters())
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in layer.parameters())
        assert sum(parameter.grad.abs().sum() for parameter in layer.parameters()) > 0


def test_same_short_cpu_schedule_is_deterministic():
    initial, cache = _fixture()
    first, second = [configure_variant(initial, "raw_slot_offset") for _ in range(2)]
    a, b = [], []
    fit(first, cache, [0, 0, 0], a.append)
    fit(second, cache, [0, 0, 0], b.append)
    assert a == b and tensor_state_sha(first) == tensor_state_sha(second)


def test_frozen_180_schedule_is_three_complete_deterministic_permutations():
    schedule = sample_schedule()
    assert len(schedule) == 540
    assert schedule == sample_schedule(180, 3, 0)
    for epoch in range(3):
        assert sorted(schedule[epoch * 180:(epoch + 1) * 180]) == list(range(180))
    assert schedule != sample_schedule(180, 3, 1)


@pytest.mark.parametrize("kwargs", [{"count": 0}, {"epochs": 0}, {"seed": -1}, {"count": True}, {"epochs": 1.5}])
def test_schedule_generator_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        sample_schedule(**kwargs)


@pytest.mark.parametrize("schedule", [[], [0, 1], [0, -1], [0, True], [0, .5]])
def test_fit_rejects_entire_invalid_schedule_before_any_update(schedule):
    initial, cache = _fixture()
    full = configure_variant(initial, "raw_slot_offset")
    before = tensor_state_sha(full)
    logs = []
    with pytest.raises(ValueError):
        fit(full, cache, schedule, logs.append)
    assert logs == [] and tensor_state_sha(full) == before


def test_valid_nonfinite_target_prevents_optimizer_update():
    initial, cache = _fixture()
    cache[0]["target"][0, 0, 0, 0] = float("nan")
    full = configure_variant(initial, "raw_slot_offset")
    before = tensor_state_sha(full)
    logs = []
    with pytest.raises(ValueError):
        fit(full, cache, [0], logs.append)
    assert tensor_state_sha(full) == before and logs == []


def test_task_or_identity_metadata_is_not_a_student_argument():
    initial, cache = _fixture()
    before_axes, before_metrics = evaluate(initial, cache)
    cache[0]["task"] = "changed_scoring_identity"
    cache[0]["teacher_identity_do_not_read"] = {"node": 123, "edge": 456}
    after_axes, after_metrics = evaluate(initial, cache)
    assert np.array_equal(before_axes, after_axes) and before_metrics == after_metrics


@pytest.mark.parametrize("issue", ["variant", "nonzero_offset"])
def test_variant_configuration_rejects_unknown_variant_or_unpaired_offsets(issue):
    initial, _ = _fixture()
    if issue == "nonzero_offset":
        with torch.no_grad():
            initial.offset.bias[0] = 1
    with pytest.raises(ValueError):
        configure_variant(initial, "unknown" if issue == "variant" else "raw_slot_offset")


def _decision_fixture():
    def rows(value):
        return [{"coordinate_mae_m": float(value), "point_mean_euclidean_m": 2. * value} for _ in range(180)]
    return rows(12), rows(10), rows(8), rows(11), [f"S{i // 18 + 1:02d}" for i in range(180)]


def test_new_fit_probe_thresholds_do_not_claim_gate_or_generalization_success():
    result = decide(*_decision_fixture())
    assert result["decision"] == "FIT_PROBE_POSITIVE"
    assert all(result["checks"].values())
    assert result["scientific_gate_pass"] is False
    assert result["generalization_claim"] is False
    assert result["detection_or_membership_accuracy_claim"] is False


@pytest.mark.parametrize("value,positive", [(9., True), (9.00001, False)])
def test_ten_percent_improvement_boundary_is_exact_not_relaxed(value, positive):
    args = _decision_fixture()
    for row in args[2]:
        row.update(coordinate_mae_m=value, point_mean_euclidean_m=2 * value)
    assert (decide(*args)["decision"] == "FIT_PROBE_POSITIVE") is positive


@pytest.mark.parametrize("improved_parents,positive", [(5, False), (6, True)])
def test_six_parent_rule_rejects_mean_gain_concentrated_in_five_parents(improved_parents, positive):
    args = _decision_fixture()
    for index, row in enumerate(args[2]):
        value = 1. if index // 18 < improved_parents else 11.
        row.update(coordinate_mae_m=value, point_mean_euclidean_m=2 * value)
    result = decide(*args)
    assert result["checks"]["coordinate_mae_at_least10pct_better_than_raw"]
    assert result["checks"]["at_least6_parent_mae_improvements"] is positive
    assert (result["decision"] == "FIT_PROBE_POSITIVE") is positive


@pytest.mark.parametrize("group,value", [(3, float("inf")), (2, float("nan")), (2, -1.)])
def test_nonfinite_or_negative_metrics_are_invalid_evidence_not_a_probe_decision(group, value):
    args = _decision_fixture()
    args[group][0]["coordinate_mae_m"] = value
    with pytest.raises(ValueError):
        decide(*args)


def test_decision_rejects_wrong_observation_or_parent_population():
    args = list(_decision_fixture())
    args[0] = args[0][:-1]
    with pytest.raises(ValueError):
        decide(*args)
    args = list(_decision_fixture())
    args[-1][0] = args[-1][-1]
    with pytest.raises(ValueError):
        decide(*args)
