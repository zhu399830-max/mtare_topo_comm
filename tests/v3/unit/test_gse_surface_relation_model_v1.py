"""CPU synthetic branch isolation, output contracts and bounded gradients."""
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from mtare_topo.representation.gse_surface_patches_v1 import extract_surface_patches
from mtare_topo.representation.gse_surface_relation_model_v1 import (
    SurfaceRelationModelV1, collate_surface_patches, REACHABILITY_ORDER,
)
from tests.v3.unit.test_gse_surface_patches_v1 import cloud


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def inputs():
    torch.manual_seed(0)
    xyz, valid, frame = cloud()
    patch = collate_surface_patches([extract_surface_patches(xyz, valid, frame)])
    return torch.tensor(xyz, dtype=torch.float32)[None], torch.randn(1, len(xyz), 128), torch.tensor(valid)[None], patch


def models():
    torch.manual_seed(11); a = SurfaceRelationModelV1("A")
    b = SurfaceRelationModelV1("B"); c = SurfaceRelationModelV1("C")
    b.load_state_dict(a.state_dict()); c.load_state_dict(a.state_dict())
    return a, b, c


def assert_equal(a, b, atol=0.):
    for field in fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if x.dtype == torch.bool: assert torch.equal(x, y)
        else: torch.testing.assert_close(x, y, rtol=0., atol=atol)


def test_independent_queries_and_multimembership_output_contract():
    x = inputs(); model = models()[2]; output = model(*x)
    assert output.anchor_position_m.shape == (1, 32, 3)
    assert output.opening_position_m.shape == (1, 64, 3)
    assert output.anchor_uncertainty_m.shape == (1, 32, 3)
    assert output.reachability_logits.shape == (1, 64, 3)
    assert REACHABILITY_ORDER == ("traversable", "blocked", "unknown")
    assert output.membership_logits.shape == output.membership_validity_logits.shape == (1, 64, 32)
    assert not torch.allclose(output.membership_logits.sigmoid().sum(-1), torch.ones(1, 64))
    assert model.anchor_queries.data_ptr() != model.opening_queries.data_ptr()
    assert output.opening_direction_valid.all() and (output.opening_dimensions_m > 0).all()


def test_abc_parameter_parity_and_true_path_isolation():
    xyz, context, valid, patch = inputs(); a, b, c = models()
    assert [sum(p.numel() for p in m.parameters()) for m in (a, b, c)].count(sum(p.numel() for p in a.parameters())) == 3
    modified = replace(patch, unary=patch.unary + .2, relation=patch.relation + .3)
    assert_equal(a(xyz, context, valid, patch), a(xyz, context, valid, modified))
    unary_result = b(xyz, context, valid, patch)
    assert_equal(unary_result, b(xyz, context, valid, replace(patch, relation=patch.relation + .3)))
    assert not torch.allclose(unary_result.opening_position_m, b(xyz, context, valid, modified).opening_position_m)
    assert not torch.allclose(c(xyz, context, valid, patch).opening_position_m,
                              c(xyz, context, valid, replace(patch, relation=patch.relation + .3)).opening_position_m)
    for model in (a, b, c):
        original = model(xyz, context, valid, patch)
        changed = model(xyz + .4, context, valid, patch)
        assert not torch.allclose(original.anchor_position_m, changed.anchor_position_m)


@pytest.mark.parametrize("branch", ["A", "B", "C"])
def test_raw_point_permutation_and_layout_bound_compact_pool(branch):
    xyz, context, valid, patch = inputs(); model = SurfaceRelationModelV1(branch)
    perm = torch.arange(xyz.shape[1] - 1, -1, -1)
    tolerance = 64 * torch.finfo(xyz.dtype).eps  # bounded floating reduction roundoff, not task scoring
    assert_equal(model(xyz, context, valid, patch), model(xyz[:, perm], context[:, perm], valid[:, perm], patch), tolerance)
    index = torch.arange(xyz.shape[1])[None] // 3
    result = model(xyz, context, valid, patch, sensor_token_index=index)
    changed = model(xyz[:, perm], context[:, perm], valid[:, perm], patch, sensor_token_index=index[:, perm])
    assert_equal(result, changed, tolerance)


def test_patch_order_permutation_transports_neighbors():
    xyz, context, valid, patch = inputs(); model = models()[2]
    p = torch.tensor([1, 0]); inverse = torch.argsort(p)
    neighbor = patch.neighbor_index[:, p]; neighbor = torch.where(neighbor >= 0, inverse[neighbor.clamp_min(0)], -1)
    swapped = replace(patch, unary=patch.unary[:, p], valid=patch.valid[:, p], neighbor_index=neighbor,
        neighbor_valid=patch.neighbor_valid[:, p], relation=patch.relation[:, p])
    assert_equal(model(xyz, context, valid, patch), model(xyz, context, valid, swapped), 1e-6)


@pytest.mark.parametrize("branch", ["A", "B", "C"])
def test_gradient_reaches_raw_and_used_patch_paths_not_frozen_context(branch):
    xyz, context, valid, patch = inputs(); context.requires_grad_(); model = SurfaceRelationModelV1(branch)
    output = model(xyz, context, valid, patch)
    loss = sum(getattr(output, f.name).square().mean() for f in fields(output) if getattr(output, f.name).dtype != torch.bool)
    loss.backward()
    assert context.grad is None
    assert model.raw_adapter[0].weight.grad.abs().sum() > 0
    if branch != "A":
        assert model.patch_adapter[0].weight.grad.abs().sum() > 0
        assert all(layer.message[0].weight.grad.abs().sum() > 0 for layer in model.patch_layers)
    else:
        assert model.patch_adapter[0].weight.grad is None
    assert model.anchor_queries.grad.abs().sum() > 0 and model.opening_queries.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_empty_and_degenerate_observations_do_not_fabricate_supported_geometry():
    xyz, context, valid, _ = inputs(); invalid = torch.zeros_like(valid)
    empty = collate_surface_patches([extract_surface_patches(xyz[0].numpy(), invalid[0].numpy(), np.zeros(xyz.shape[1], dtype=np.int64))])
    for model in models():
        output = model(xyz, context, invalid, empty)
        assert not output.observation_supported.any() and not output.opening_direction_valid.any()
        assert all(not getattr(output, f.name).any() for f in fields(output))
    singleton = extract_surface_patches(np.array([[1., 0., 0.]]), np.array([True]), np.array([0]))
    patches = collate_surface_patches([singleton])
    output = models()[2](torch.tensor([[[1., 0., 0.]]]), torch.zeros(1, 1, 128), torch.ones(1, 1, dtype=torch.bool), patches)
    assert output.observation_supported.all() and patches.valid.all()


def test_no_teacher_keyword_and_unsafe_large_raw_memory_rejected():
    x = inputs(); model = models()[2]
    with pytest.raises(TypeError): model(*x, teacher={"center": [0, 0, 0]})
    with pytest.raises(ValueError, match="compact"):
        model(torch.zeros(1, 901, 3), torch.zeros(1, 901, 128), torch.ones(1, 901, dtype=torch.bool), x[-1])
    patch = replace(x[-1], neighbor_index=torch.full_like(x[-1].neighbor_index, 999))
    with pytest.raises(ValueError, match="indices"):
        model(*x[:3], patch)


def test_relation_layers_singleton_bc_forward_identical():
    xyz = np.array([[1., 0., 0.]])
    patch = collate_surface_patches([extract_surface_patches(xyz, np.array([True]), np.array([0]))])
    _, b, c = models()
    args = (torch.tensor(xyz, dtype=torch.float32)[None], torch.zeros(1, 1, 128), torch.ones(1, 1, dtype=torch.bool), patch)
    assert_equal(b(*args), c(*args))


def test_float64_permutation_contract_and_repeat_determinism():
    xyz, context, valid, patch = inputs(); xyz = xyz.double(); context = context.double()
    patch = replace(patch, unary=patch.unary.double(), relation=patch.relation.double())
    model = SurfaceRelationModelV1("C").double()
    index = torch.arange(xyz.shape[1])[None] // 3; perm = torch.arange(xyz.shape[1] - 1, -1, -1)
    original = model(xyz, context, valid, patch, sensor_token_index=index)
    assert_equal(original, model(xyz, context, valid, patch, sensor_token_index=index))
    assert_equal(original, model(xyz[:, perm], context[:, perm], valid[:, perm], patch,
                                 sensor_token_index=index[:, perm]), 64 * torch.finfo(torch.float64).eps)


def test_existing_compact57600point_interface_without_loading_backbone():
    from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import CompactFrozenDualPathFeatures
    points = torch.zeros(1, 57600, 3); points[..., 2] = 1.
    context = torch.zeros(1, 900, 128); valid = torch.ones(1, 57600, dtype=torch.bool)
    t = torch.arange(5)[:, None, None]; azimuth = torch.arange(720)[None, None]
    index = (t * 180 + azimuth // 4).expand(5, 16, 720).reshape(1, -1)
    compact = CompactFrozenDualPathFeatures(points, context, valid, index)
    patches = collate_surface_patches([extract_surface_patches(points[0].numpy(), valid[0].numpy(),
        np.repeat(np.arange(5), 16 * 720))])
    model = SurfaceRelationModelV1("C")
    output = model.forward_compact(compact, patches)
    assert output.anchor_position_m.shape == (1, 32, 3) and output.observation_supported.all()
    output.opening_position_m.square().mean().backward()
    assert torch.isfinite(model.raw_adapter[0].weight.grad).all()
