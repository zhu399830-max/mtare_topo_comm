import numpy as np
import pytest
import torch

from evaluate_gse_joint_cyclic_gap_simplex_selection_v1 import _ensemble
from execute_gse_joint_cyclic_gap_simplex_readiness_v2 import _synthetic_contract_v2
from train_gse_circular_peak_geometry_v1 import _clip_finite_gradients

from mtare_topo.representation.gse_joint_cyclic_gap_simplex import (
    circular_gaps_from_ordered_bearings,
    decode_cyclic_bearings,
    periodic_linear_coordinates,
    periodic_linear_sample,
    positive_gap_simplex,
    stable_phase_atan2,
)


def test_single_exit_gap_closes_full_circle() -> None:
    bearing = torch.tensor([[17.0], [359.5]], dtype=torch.float64)
    expected = torch.full_like(bearing, 360.0)
    torch.testing.assert_close(circular_gaps_from_ordered_bearings(bearing), expected)


def test_phase_angle_has_finite_zero_resultant_gradient() -> None:
    y = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    stable_phase_atan2(y, x).sum().backward()
    assert bool(torch.isfinite(y.grad).all())
    assert bool(torch.isfinite(x.grad).all())
    torch.testing.assert_close(y.grad, torch.zeros_like(y))
    torch.testing.assert_close(x.grad, torch.zeros_like(x))


def test_phase_angle_matches_atan2_away_from_singularity() -> None:
    y = torch.tensor([0.2, -0.4, 1.0], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([0.7, 0.3, -0.2], dtype=torch.float64, requires_grad=True)
    actual = stable_phase_atan2(y, x)
    torch.testing.assert_close(actual, torch.atan2(y, x))
    actual.sum().backward()
    denominator = x.detach().square() + y.detach().square()
    torch.testing.assert_close(y.grad, x.detach() / denominator)
    torch.testing.assert_close(x.grad, -y.detach() / denominator)


def test_periodic_sampler_rejects_nonfinite_bearing() -> None:
    directional = torch.zeros((1, 2, 180), dtype=torch.float32)
    with np.testing.assert_raises(FloatingPointError):
        periodic_linear_sample(directional, torch.tensor([[float("nan")]]))


def test_periodic_index_canonicalises_negative_roundoff_at_zero() -> None:
    # Exact value attributed in the formal seed-1 CUDA diagnostic.
    bearing = torch.tensor([-2.842170943040401e-14], dtype=torch.float64)
    fraction, lower, upper = periodic_linear_coordinates(bearing)
    assert int(lower.item()) == 0
    assert int(upper.item()) == 1
    assert float(fraction.item()) == 0.0
    directional = torch.arange(180, dtype=torch.float64)[None, None]
    sampled = periodic_linear_sample(directional, bearing[None])
    torch.testing.assert_close(sampled, torch.zeros_like(sampled))


def test_training_rejects_nonfinite_gradient_before_optimizer_step() -> None:
    model = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        _clip_finite_gradients(model)


def test_gap_simplex_is_positive_and_closes_circle():
    logits = torch.tensor([[1.0, -2.0, 0.5, 4.0]])
    gaps = positive_gap_simplex(logits)
    assert bool((gaps > 0).all())
    assert torch.allclose(gaps.sum(-1), torch.ones(1))
    bearings = decode_cyclic_bearings(torch.tensor([350.0]), gaps)
    recovered = circular_gaps_from_ordered_bearings(bearings)
    assert torch.allclose(recovered, gaps * 360.0, atol=2e-5)


def test_decode_handles_wrap_and_known_gaps():
    gaps = torch.tensor([[20.0, 90.0, 250.0]]) / 360.0
    bearings = decode_cyclic_bearings(torch.tensor([350.0]), gaps)
    assert torch.allclose(bearings, torch.tensor([[350.0, 10.0, 100.0]]), atol=1e-5)


def test_periodic_linear_sample_is_rotation_equivariant():
    directional = torch.arange(180, dtype=torch.float32)[None, None]
    bearings = torch.tensor([[0.0, 20.0, 358.0]])
    base = periodic_linear_sample(directional, bearings)
    shifted = periodic_linear_sample(torch.roll(directional, 10, -1), torch.remainder(bearings + 20.0, 360.0))
    assert torch.equal(base, shifted)


def test_gap_decoder_has_finite_nonzero_gradient():
    logits = torch.tensor([[0.2, -0.1, 0.4]], requires_grad=True)
    bearings = decode_cyclic_bearings(torch.tensor([12.0]), positive_gap_simplex(logits))
    loss = ((bearings - torch.tensor([[12.0, 95.0, 241.0]])) ** 2).mean()
    loss.backward()
    assert np.isfinite(logits.grad.numpy()).all()
    assert float(logits.grad.abs().sum()) > 0


def test_proper_phase_likelihood_and_gap_corrective_are_directional():
    diagnostic = _synthetic_contract_v2()
    assert diagnostic["correct_phase_likelihood"] < diagnostic["uniform_phase_likelihood"]
    assert diagnostic["uniform_phase_likelihood"] < diagnostic["wrong_phase_likelihood"]
    assert diagnostic["uniform_target_bin_gradient"] < 0
    assert diagnostic["collapsed_localization"] > diagnostic["correct_localization"]
    assert diagnostic["collapsed_gap_gradient_l1"] > 0


def test_ensemble_aligns_cyclic_starts_and_penalizes_phase_uncertainty():
    data = {
        "presence": np.zeros((1, 180), dtype=np.uint8),
        "count_probability": np.asarray([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
    }
    branches = (np.asarray([10.0, 100.0, 240.0]), np.asarray([100.0, 240.0, 10.0]), np.asarray([240.0, 10.0, 100.0]))
    for seed, bearings in enumerate(branches):
        full = np.zeros((1, 10), dtype=np.float32)
        full[0, 3:6] = bearings
        data[f"seed{seed}_joint_bearing_deg"] = full
        data[f"seed{seed}_joint_bearing_scale_deg"] = np.ones((1, 10), dtype=np.float32) * 2.0
        data[f"seed{seed}_phase_concentration"] = np.ones((1, 4), dtype=np.float32) * 0.5
        data[f"seed{seed}_exit_opening_width_m"] = np.ones((1, 10), dtype=np.float32)
        data[f"seed{seed}_exit_vertical_profile_m"] = np.zeros((1, 10, 4), dtype=np.float32)
    decoded = _ensemble(data)
    assert np.allclose(np.sort(decoded["bearing"][0, :3]), [10.0, 100.0, 240.0], atol=1e-5)
    assert 0 < decoded["score"][0] < 0.5
