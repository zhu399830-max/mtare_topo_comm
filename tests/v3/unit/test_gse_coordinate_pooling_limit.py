"""Constructive limits of the CURRENT coordinate readout, not new accuracy.

These synthetic coordinate clouds exercise the production _token_xyz stage;
they are not claims that a particular recorded LiDAR scan has this geometry.
Even an encoder that recognizes two layers cannot undo a coordinate convex
hull restriction through its choice of softmax weights.
"""
import torch

from mtare_topo.representation.primitive_relation_model import _token_xyz


def _two_layers(height=3.):
    points = torch.zeros((1, 5, 16, 720, 3), dtype=torch.float64)
    points[..., 0] = torch.linspace(-10, 10, 720, dtype=torch.float64)
    points[:, :, :8, :, 2] = -height
    points[:, :, 8:, :, 2] = height
    valid = torch.ones((1, 5, 16, 720), dtype=torch.bool)
    return points, valid


def test_current_pool_flattens_symmetric_layers_even_with_all_returns_valid():
    points, valid = _two_layers()
    pooled, pooled_valid = _token_xyz(points, valid)
    assert bool(pooled_valid.all())
    assert torch.count_nonzero(pooled[..., 2]) == 0
    assert torch.unique(points[..., 2]).tolist() == [-3., 3.]
    other, _ = _two_layers(6.)
    # Same geometry-coordinate support pool, not necessarily same CNN features.
    assert torch.equal(_token_xyz(other, valid)[0], pooled)


def test_no_softmax_attention_training_can_recover_height_outside_pooled_plane():
    points, valid = _two_layers()
    memory, _ = _token_xyz(points, valid)
    memory = memory.reshape(1, 900, 3)
    logits = torch.linspace(-4, 4, 5400, dtype=torch.float64).reshape(1, 2, 3, 900).requires_grad_()
    axis = torch.einsum("bscn,bnd->bscd", torch.softmax(logits, -1), memory)
    expected_height = torch.tensor([-3., 3.], dtype=torch.float64)[None, :, None]
    loss = (axis[..., 2] - expected_height).square().mean()
    assert loss.item() == 9.
    gradient, = torch.autograd.grad(loss, logits)
    assert torch.count_nonzero(gradient) == 0


def test_unpooled_coordinates_retain_oracle_support_for_both_layers():
    points, _ = _two_layers()
    # A constructive witness only: these choices use synthetic layer identity.
    # A deployment head would need to LEARN membership, not receive this oracle.
    witness = torch.stack((points[0, 0, 0, [0, 360, 719]], points[0, 0, 8, [0, 360, 719]]))
    assert torch.equal(witness[..., 2], torch.tensor([[-3.] * 3, [3.] * 3], dtype=torch.float64))
    assert torch.equal(witness[0, :, 0], witness[1, :, 0])


def test_same_averaging_can_remove_separate_depth_support_not_only_height():
    points, valid = _two_layers()
    points.zero_()
    points[:, :, :8, :, 0] = 10.
    points[:, :, 8:, :, 0] = 30.
    pooled, _ = _token_xyz(points, valid)
    assert torch.equal(pooled[..., 0], torch.full((1, 5, 180), 20., dtype=torch.float64))
    # Any nonnegative normalized readout of identical x=20 support stays x=20.
    logits = torch.zeros((2, 900), dtype=torch.float64)
    result = torch.softmax(logits, -1) @ pooled.reshape(900, 3)
    assert torch.allclose(result[:, 0], torch.tensor([20., 20.], dtype=torch.float64))
    assert torch.allclose((result[:, 0] - torch.tensor([10., 30.])).abs(), torch.tensor([10., 10.], dtype=torch.float64))
