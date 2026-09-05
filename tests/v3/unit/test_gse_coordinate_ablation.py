"""Synthetic coordinate intervention only: no data, checkpoints or optimizer."""
import pytest
import torch
from mtare_topo.representation.gse_coordinate_ablation import coordinate_inputs
from mtare_topo.representation.gse_point_axis_readout import PointAxisReadout
from mtare_topo.representation.gse_point_axis_probe import configure_variant


def inputs():
    generator = torch.Generator().manual_seed(19)
    points = torch.tensor([[[-2., 0., -3.], [2., 0., 3.], [4., 0., -2.], [6., 0., 2.]]])
    return (points, torch.ones(1, 4, dtype=torch.bool),
            torch.randn(1, 2, 8, generator=generator),
            torch.tensor([[[0., 0., 0.], [5., 0., 0.]]]),
            torch.tensor([[0, 0, 1, 1]]), torch.randn(1, 3, 8, generator=generator))


def test_only_coordinates_change_and_inputs_are_not_mutated():
    original = inputs(); saved = [t.clone() for t in original]
    raw = coordinate_inputs(original, "raw_coordinates")
    mean = coordinate_inputs(original, "mean_broadcast_coordinates")
    assert all(a is b for a, b in zip(original, raw))
    assert all(a is b for a, b in zip(original[1:], mean[1:]))
    torch.testing.assert_close(mean[0], original[3][:, [0, 0, 1, 1]])
    assert mean[0].shape == original[0].shape
    assert all(torch.equal(a, b) for a, b in zip(original, saved))


def test_mean_control_erases_within_token_layers_without_changing_features():
    original = inputs(); changed = list(original)
    changed[0] = original[0].clone(); changed[0][..., 2] *= 2
    assert not torch.equal(coordinate_inputs(original, "raw_coordinates")[0],
                           coordinate_inputs(changed, "raw_coordinates")[0])
    assert all(torch.equal(a, b) for a, b in zip(
        coordinate_inputs(original, "mean_broadcast_coordinates"),
        coordinate_inputs(changed, "mean_broadcast_coordinates")))


def test_input_at_means_has_exact_output_parity_and_equal_parameter_sets():
    original = list(inputs()); original[0] = original[3][:, [0, 0, 1, 1]].clone()
    torch.manual_seed(5)
    initial = PointAxisReadout(model_dim=8, point_dim=4)
    raw = configure_variant(initial, "raw_no_offset")
    mean = configure_variant(initial, "raw_no_offset")
    assert all(torch.equal(a, b) for a, b in zip(raw.parameters(), mean.parameters()))
    for head, variant in ((raw, "raw_coordinates"), (mean, "mean_broadcast_coordinates")):
        output = head(*coordinate_inputs(original, variant)).votes.axis_control_m
        output.square().sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in head.parameters() if p.requires_grad)
        assert all(p.grad is None and not bool(p.any())
                   for layer in (head.offset, head.slot_offset) for p in layer.parameters())
    assert torch.equal(raw(*coordinate_inputs(original, "raw_coordinates")).votes.axis_control_m,
                       mean(*coordinate_inputs(original, "mean_broadcast_coordinates")).votes.axis_control_m)


def test_invalid_padding_does_not_add_or_remove_valid_returns():
    original = list(inputs()); original[1][0, 0] = False; original[0][0, 0] = float("nan")
    mean = coordinate_inputs(original, "mean_broadcast_coordinates")
    assert mean[1] is original[1] and int(mean[1].sum()) == 3
    assert torch.equal(mean[0][0, 0], torch.zeros(3))


@pytest.mark.parametrize("defect", ["variant", "targets", "empty", "bad_mask", "bad_index", "no_evidence", "nan", "dtype"])
def test_invalid_contracts_fail_closed(defect):
    original = list(inputs()); variant = "raw_coordinates"
    if defect == "variant": variant = "unknown"
    elif defect == "targets": original.append(torch.zeros(1))
    elif defect == "empty": original[0] = original[0][:, :0]
    elif defect == "bad_mask": original[1] = original[1].float()
    elif defect == "bad_index": original[4][0, 0] = 2
    elif defect == "no_evidence": original[1].fill_(False)
    elif defect == "nan": original[0][0, 0, 0] = float("nan")
    elif defect == "dtype": original[2] = original[2].double()
    with pytest.raises(ValueError): coordinate_inputs(original, variant)
