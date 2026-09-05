"""Synthetic neural-interface checks, not data fitting or learned accuracy."""
import pytest
import torch
from torch import nn

from mtare_topo.representation.gse_point_axis_readout import (
    PointAxisReadout, FrozenBackbonePointAxisAdapter,
)


def _fixture():
    torch.manual_seed(19)
    head = PointAxisReadout(model_dim=8, point_dim=4).double()
    points = torch.randn(2, 13, 3, dtype=torch.float64)
    valid = torch.ones(2, 13, dtype=torch.bool)
    memory = torch.randn(2, 3, 8, dtype=torch.float64)
    xyz = torch.randn(2, 3, 3, dtype=torch.float64)
    index = torch.arange(13).remainder(3)[None].expand(2, -1)
    slots = torch.randn(2, 2, 8, dtype=torch.float64)
    return head, [points, valid, memory, xyz, index, slots]


def test_init_is_raw_coordinate_vote_not_old_mean_and_all_parameters_receive_gradient():
    head, inputs = _fixture()
    out = head(*inputs)
    assert out.surface_to_axis_offset_m.shape == (2, 2, 13, 3)
    assert not bool(out.surface_to_axis_offset_m.any())
    expected = torch.einsum("bscn,bnd->bscd", out.votes.control_weights, inputs[0])
    torch.testing.assert_close(out.votes.axis_control_m, expected)
    target = torch.randn_like(expected)
    (out.votes.axis_control_m - target).square().mean().backward()
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert bool(parameter.grad.abs().sum() > 0), name


def test_same_parent_sensor_token_points_are_distinguishable():
    head, inputs = _fixture()
    # Make parent contexts identical. Distinct XYZ must still affect membership.
    inputs[4] = torch.zeros_like(inputs[4])
    out = head(*inputs)
    assert not torch.allclose(out.membership_logits[..., 0], out.membership_logits[..., 1])


def test_point_and_slot_permutation_including_conditioned_offset():
    head, inputs = _fixture()
    with torch.no_grad():
        head.slot_offset.weight.fill_(.03)
    first = head(*inputs)
    permutation = torch.randperm(13)
    permuted = list(inputs)
    for i in (0, 1, 4):
        permuted[i] = inputs[i][:, permutation]
    torch.testing.assert_close(head(*permuted).votes.axis_control_m, first.votes.axis_control_m)
    permuted = list(inputs)
    permuted[-1] = inputs[-1][:, [1, 0]]
    other = head(*permuted)
    torch.testing.assert_close(other.votes.axis_control_m, first.votes.axis_control_m[:, [1, 0]])
    torch.testing.assert_close(other.surface_to_axis_offset_m, first.surface_to_axis_offset_m[:, [1, 0]])
    torch.testing.assert_close(other.membership_logits[:, -1], first.membership_logits[:, -1])


def test_repeat_and_invalid_nan_padding_have_no_effect():
    head, inputs = _fixture()
    inputs[1][:, -1] = False
    first = head(*inputs)
    inputs[0][:, -1] = torch.nan
    other = head(*inputs)
    assert torch.equal(first.votes.axis_control_m, other.votes.axis_control_m)
    assert torch.equal(other.votes.axis_control_m, head(*inputs).votes.axis_control_m)
    assert not bool(other.surface_to_axis_offset_m[:, :, -1].any())


@pytest.mark.parametrize("field", [0, 2, 3, 5])
def test_valid_nonfinite_floating_inputs_fail(field):
    head, inputs = _fixture()
    inputs[field].flatten()[0] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        head(*inputs)


@pytest.mark.parametrize("index", [-1, 3])
def test_bad_sensor_token_index_fails(index):
    head, inputs = _fixture()
    inputs[4] = inputs[4].clone()
    inputs[4][0, 0] = index
    with pytest.raises(ValueError, match="bounds"):
        head(*inputs)


def test_empty_evidence_does_not_fabricate_geometry():
    head, inputs = _fixture()
    inputs[1][0] = False
    with pytest.raises(ValueError, match="all-invalid"):
        head(*inputs)


class _Decoder(nn.Module):
    def forward(self, query, memory, memory_key_padding_mask):
        assert not torch.is_grad_enabled()
        return query + memory.mean(dim=1, keepdim=True)


class _FrozenDummy(nn.Module):
    """Tests adapter plumbing only; not a replacement trained backbone."""
    def __init__(self):
        super().__init__()
        self.slot_query = nn.Parameter(torch.randn(2, 8))
        self.context = nn.Parameter(torch.randn(1, 900, 8))
        self.slot_decoder = _Decoder()

    def _memory(self, scans, translation, yaw):
        assert not self.training and not torch.is_grad_enabled()
        b = len(scans)
        return (self.context.expand(b, -1, -1), torch.ones(b, 900, dtype=torch.bool),
                torch.zeros(b, 900, 3), None)


def test_causal_adapter_freezes_backbone_and_uses_sensor_not_teacher_indices():
    torch.manual_seed(23)
    backbone = _FrozenDummy()
    original = {k: v.clone() for k, v in backbone.state_dict().items()}
    adapter = FrozenBackbonePointAxisAdapter(backbone, PointAxisReadout(8, 4)).train()
    assert not backbone.training and adapter.readout.training
    seen = {}
    def capture(module, args):
        seen["indices"] = args[4]
        seen["points"] = args[0]
    handle = adapter.readout.register_forward_pre_hook(capture)
    scans = torch.zeros(1, 5, 2, 16, 720)
    scans[:, :, 0] = .1
    scans[:, :, 1] = 1.
    output = adapter(scans, torch.zeros(1, 5, 3), torch.zeros(1, 5))
    output.votes.axis_control_m.square().mean().backward()
    handle.remove()
    assert seen["points"].shape == (1, 57600, 3)
    indices = seen["indices"].reshape(5, 16, 720)
    for frame, elevation, column in ((0, 0, 0), (0, 15, 719), (4, 9, 301)):
        assert indices[frame, elevation, column] == frame * 180 + column // 4
    for name, value in backbone.state_dict().items():
        assert torch.equal(value, original[name])
    assert all(p.grad is None and not p.requires_grad for p in backbone.parameters())
    assert adapter.readout.slot_offset.weight.grad.abs().sum() > 0
    assert output.votes.axis_control_m.shape == (1, 2, 3, 3)


def test_forward_does_not_accept_teacher_or_identity_arguments():
    head, inputs = _fixture()
    with pytest.raises(TypeError):
        head(*inputs, primitive_identity=torch.zeros(2, 13))


def test_full_size_real_backbone_adapter_forward_backward_without_checkpoint_or_optimizer():
    from mtare_topo.representation.primitive_relation_observable_model import ObservableSparsePortRelationNet
    torch.manual_seed(29)
    backbone = ObservableSparsePortRelationNet()
    before = {k: v.clone() for k, v in backbone.state_dict().items()}
    adapter = FrozenBackbonePointAxisAdapter(backbone).train()
    scans = torch.zeros(1, 5, 2, 16, 720)
    scans[:, :, 0] = torch.linspace(.08, .22, 720)[None, None, None]
    scans[:, :, 1] = 1.
    translation = torch.zeros(1, 5, 3)
    translation[0, :, 0] = torch.arange(-4., 1.)
    output = adapter(scans, translation, torch.zeros(1, 5))
    assert output.votes.control_weights.shape == (1, 32, 3, 57600)
    assert output.surface_to_axis_offset_m.shape == (1, 32, 57600, 3)
    assert sum(p.numel() for p in adapter.readout.parameters()) < 50000
    (output.votes.axis_control_m - torch.tensor([1., 2., 3.])).square().mean().backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in adapter.readout.parameters())
    assert adapter.readout.slot_offset.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in backbone.parameters())
    assert all(torch.equal(v, before[k]) for k, v in backbone.state_dict().items())
