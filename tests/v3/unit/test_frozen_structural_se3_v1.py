"""Synthetic SE3 wiring; zero actual records/checkpoints, no optimizer."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from mtare_topo.integration.frozen_structural_se3_v1 import (
    FrozenSE3EncoderAdapterV1, FrozenStructuralSE3TokenExtractor,
    prepare_common_observation_se3, relative_sensor_motion_se3,
)
from mtare_topo.integration.frozen_structural_tokens import FrozenStructuralTokenExtractor
from mtare_topo.representation.branch_relation_learning import RayBranchRelationModel
from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import FrozenDualPathEncoderAdapterV1
from mtare_topo.representation.gse_structure_context import CausalFrameOrderContext
from mtare_topo.representation.gse_surface_patches_v1 import extract_surface_patches
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points, _token_xyz


class SyntheticBackbone(nn.Module):
    def __init__(self, drop_rotation=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.25))
        self.drop_rotation = drop_rotation
        self.calls = []

    def _memory(self, *inputs, relative_rotation_current_sensor=None):
        self.calls.append((self.training, torch.is_grad_enabled(), relative_rotation_current_sensor))
        use = None if self.drop_rotation else relative_rotation_current_sensor
        xyz, valid = register_causal_lidar_points(*inputs, relative_rotation_current_sensor=use)
        means, mask = _token_xyz(xyz, valid)
        b = len(xyz)
        memory = means.reshape(b, 900, 3).repeat(1, 1, 43)[..., :128] + self.scale
        return memory, mask.reshape(b, 900), means.reshape(b, 900, 3), memory.reshape(b, 5, 180, 128)


def rotation(roll, pitch, yaw):
    r, p, y = np.deg2rad([roll, pitch, yaw])
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def inputs():
    rv = torch.zeros(1, 5, 2, 16, 720)
    for ring, az in ((0, 4), (15, 5), (8, 700)):
        rv[:, :, 0, ring, az] = .08
        rv[:, :, 1, ring, az] = 1
    poses = np.tile(np.eye(4), (5, 1, 1))
    for i in range(5):
        poses[i, :3, :3] = rotation(i*7-8, i*3-5, i*9)
        poses[i, :3, 3] = [i*.1, i*-.08, i*.2]
    t, y, r = relative_sensor_motion_se3(poses)
    return rv, torch.from_numpy(t)[None], torch.from_numpy(y)[None], torch.from_numpy(r.astype(np.float32))[None], poses


def student(rv, t, y):
    return SimpleNamespace(ranges_m=(rv[0, :, 0]*50).numpy(), valid_mask=rv[0, :, 1].numpy().astype(np.uint8),
        relative_translation_current_sensor_m=t[0].numpy(), relative_yaw_current_sensor_deg=y[0].numpy())


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_arbitrary_rigid_pose_direct_coordinate_and_common_world_transform():
    rv, t, y, r, poses = inputs()
    original = poses.copy()
    adapter = FrozenSE3EncoderAdapterV1(SyntheticBackbone(), nn.Identity())
    result = adapter.extract_compact_features(rv, t, y, relative_rotation_current_sensor=r)
    local, mask = register_causal_lidar_points(rv, torch.zeros_like(t), torch.zeros_like(y))
    expected = np.zeros((5, 16, 720, 3))
    for i in range(5):
        transform = np.linalg.inv(poses[-1]) @ poses[i]
        expected[i] = local.numpy()[0, i] @ transform[:3, :3].T + transform[:3, 3]
    expected[~mask.numpy()[0]] = 0
    np.testing.assert_allclose(result.points_xyz_m[0].numpy(), expected.reshape(57600, 3), atol=2e-6, rtol=1e-6)
    q = np.eye(4); q[:3, :3] = rotation(40, -20, 80); q[:3, 3] = [100, -25, 4]
    transformed = q[None] @ poses
    t2, y2, r2 = relative_sensor_motion_se3(transformed)
    np.testing.assert_allclose(t2, t[0].numpy(), atol=1e-6)
    np.testing.assert_allclose(y2, y[0].numpy(), atol=1e-5)
    np.testing.assert_allclose(r2, r[0].numpy(), atol=1e-7)
    np.testing.assert_array_equal(poses, original)
    assert adapter.backbone.calls[0][:2] == (False, False)
    assert torch.equal(adapter.backbone.calls[0][2], r)
    assert not result.frozen_sensor_context.requires_grad
    assert result.valid.sum() == 15 and result.sensor_token_index.shape == (1, 57600)


def test_backbone_dropping_rotation_cannot_supply_mismatched_features():
    rv, t, y, r, _ = inputs()
    adapter = FrozenSE3EncoderAdapterV1(SyntheticBackbone(drop_rotation=True), nn.Identity())
    with pytest.raises(ValueError, match='coordinates disagree'):
        adapter.extract_compact_features(rv, t, y, relative_rotation_current_sensor=r)


@pytest.mark.parametrize('problem', ['reflection', 'nonorthogonal', 'current', 'nonfinite', 'dtype', 'yaw', 'shape'])
def test_invalid_rotation_or_embedding_rejected_before_backbone(problem):
    rv, t, y, r, _ = inputs()
    if problem == 'reflection': r[:, 0, :, 0] *= -1
    elif problem == 'nonorthogonal': r[:, 0] *= 2
    elif problem == 'current': r[:, -1] = r[:, 0]
    elif problem == 'nonfinite': r[:, 0, 0, 0] = float('nan')
    elif problem == 'dtype': r = r.double()
    elif problem == 'yaw': y[:, 0] += 10
    elif problem == 'shape': r = r[:, :4]
    backbone = SyntheticBackbone(); adapter = FrozenSE3EncoderAdapterV1(backbone, nn.Identity())
    with pytest.raises(ValueError): adapter.extract_compact_features(rv, t, y, relative_rotation_current_sensor=r)
    assert backbone.calls == []


def test_rotation_mandatory_empty_history_preserved_and_unfreeze_rejected():
    rv, t, y, r, _ = inputs()
    adapter = FrozenSE3EncoderAdapterV1(SyntheticBackbone(), nn.Identity())
    with pytest.raises(TypeError): adapter.extract_compact_features(rv, t, y)
    empty = rv.clone(); empty[:, :, 1] = 0
    result = adapter.extract_compact_features(empty, t, y, relative_rotation_current_sensor=r)
    assert not result.valid.any() and not result.frozen_sensor_context.any()
    assert adapter.backbone.calls == []
    empty[:, 0, 1, 0, 4] = 1
    with pytest.raises(ValueError, match='partial empty history'):
        adapter.extract_compact_features(empty, t, y, relative_rotation_current_sensor=r)
    adapter.backbone.requires_grad_(True)
    with pytest.raises(ValueError, match='unfrozen'):
        adapter.extract_compact_features(rv, t, y, relative_rotation_current_sensor=r)


def test_direct_adapter_current_identity_is_exact_not_allclose():
    rv, t, y, r, _ = inputs()
    r[:, -1, 0, 2] = 1e-10
    assert torch.allclose(r[:, -1], torch.eye(3)[None], atol=1e-6, rtol=0)
    backbone = SyntheticBackbone(); adapter = FrozenSE3EncoderAdapterV1(backbone, nn.Identity())
    with pytest.raises(ValueError, match='exact current identity'):
        adapter.extract_compact_features(rv, t, y, relative_rotation_current_sensor=r)
    assert backbone.calls == []


def test_pure_yaw_compact_parity_and_identity_c_tokens_no_new_parameters():
    rv, t, y, r, _ = inputs()
    angles = np.array([-30, 12, 90, -4, 0], dtype=np.float32)
    y = torch.from_numpy(angles)[None]
    r = torch.from_numpy(np.stack([rotation(0, 0, a) for a in angles]).astype(np.float32))[None]
    old = FrozenDualPathEncoderAdapterV1(SyntheticBackbone(), nn.Identity())
    new = FrozenSE3EncoderAdapterV1(SyntheticBackbone(), nn.Identity())
    a = old.extract_compact_features(rv, t, y)
    b = new.extract_compact_features(rv, t, y, relative_rotation_current_sensor=r)
    assert torch.equal(a.valid, b.valid) and torch.equal(a.sensor_token_index, b.sensor_token_index)
    torch.testing.assert_close(a.points_xyz_m, b.points_xyz_m, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(a.frozen_sensor_context, b.frozen_sensor_context, atol=1e-6, rtol=1e-6)
    assert old.state_dict().keys() == new.state_dict().keys()
    # Identity is also exact at the complete C token interface. Nonidentity
    # floating yaw parity is not a promise of identical boundary voxelization.
    y.zero_(); t.zero_(); r[:] = torch.eye(3)
    model = RayBranchRelationModel('C'); original = copy.deepcopy(model.state_dict())
    first = FrozenStructuralTokenExtractor(old, copy.deepcopy(model))
    second = FrozenStructuralSE3TokenExtractor(new, model)
    context = CausalFrameOrderContext('sensor_current', (0, 1, 2, 3, 4), 4, 'synthetic')
    refs = tuple(str(i) for i in range(5)); s = student(rv, t, y)
    a = first.extract(s, context=context, source_refs=refs)
    b = second.extract(s, context=context, source_refs=refs, relative_rotation_current_sensor=r[0].numpy())
    np.testing.assert_allclose(a.ray_tokens, b.ray_tokens, atol=1e-6, rtol=1e-6)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


def test_non_yaw_common_reextracts_geometry_and_does_not_use_old_cache():
    rv, t, y, r, _ = inputs()
    adapter = FrozenSE3EncoderAdapterV1(SyntheticBackbone(), nn.Identity())
    obs = prepare_common_observation_se3(adapter, student(rv, t, y), relative_rotation_current_sensor=r[0].numpy(), device='cpu')
    expected, mask = register_causal_lidar_points(rv, t, y, relative_rotation_current_sensor=r)
    np.testing.assert_array_equal(obs.registered_returns_xyz_m, expected[0].reshape(-1, 3).numpy())
    assert obs.surface_patches.point_count.sum() == int(mask.sum())
    assert obs.surface_patches.voxel_size_m == .5 and obs.surface_patches.roi_radius_m == 10
    assert obs.full_sensor_context.shape == (900, 128)
    assert len(obs.surface_patches.centers_m) <= 4096


def test_full_token_non_yaw_and_future_context_fail_closed():
    rv, t, y, r, _ = inputs()
    model = RayBranchRelationModel('C')
    extractor = FrozenStructuralSE3TokenExtractor(FrozenSE3EncoderAdapterV1(SyntheticBackbone(), nn.Identity()), model)
    context = CausalFrameOrderContext('sensor_current', (0, 1, 2, 3, 4), 4, 'synthetic')
    refs = tuple('source-'+str(i) for i in range(5)); s = student(rv, t, y)
    output = extractor.extract(s, context=context, source_refs=refs, relative_rotation_current_sensor=r[0].numpy())
    assert output.ray_tokens.shape == (3, 128) and output.source_refs == refs
    assert output.geometric_support.all()
    with pytest.raises(ValueError): output.ray_tokens.setflags(write=True)
    with pytest.raises(ValueError): replace(context, source_frame_ids=(0, 1, 2, 3, 5))
    with pytest.raises(ValueError):
        extractor.extract(s, context=replace(context, observation_frame_id=5), source_refs=refs, relative_rotation_current_sensor=r[0].numpy())
    with pytest.raises(ValueError, match='source references'):
        extractor.extract(s, context=context, source_refs=('duplicated-source',)*5, relative_rotation_current_sensor=r[0].numpy())
    bad = r[0].double().numpy(); bad[-1, 0, 0] += 1e-10
    with pytest.raises(ValueError, match='before dtype conversion'):
        extractor.extract(s, context=context, source_refs=refs, relative_rotation_current_sensor=bad)


def test_voxel_boundary_changes_are_visible_not_false_exact_equivariance():
    xyz = np.array([[.25, 1., 1.], [.5, 1., 1.]], dtype=np.float32)
    valid = np.ones(2, dtype=bool); frames = np.array([0, 4])
    original = extract_surface_patches(xyz, valid, frames)
    changed = xyz.copy(); changed[1, 0] = np.nextafter(np.float32(.5), np.float32(0))
    perturbed = extract_surface_patches(changed, valid, frames)
    assert len(original.centers_m) == 2 and len(perturbed.centers_m) == 1
    assert original.point_count.sum() == perturbed.point_count.sum() == 2


def test_full_rotation_unknown_empty_roi_keeps_all_far_rays_then_empty_observation():
    rv, t, y, r, _ = inputs()
    rv[:, :, 0][rv[:, :, 1].bool()] = .4
    extractor = FrozenStructuralSE3TokenExtractor(
        FrozenSE3EncoderAdapterV1(SyntheticBackbone(), nn.Identity()), RayBranchRelationModel('C'))
    context = CausalFrameOrderContext('sensor_current', (0, 1, 2, 3, 4), 4, 'synthetic')
    refs = tuple('raw-'+str(i) for i in range(5))
    result = extractor.extract(student(rv, t, y), context=context, source_refs=refs,
                               relative_rotation_current_sensor=r[0].numpy())
    assert result.ray_tokens.shape == (3, 128)
    assert result.patch_centers_current_sensor_m.shape == (0, 3)
    assert result.nearest_patch_indices.shape == (3, 0) and not result.geometric_support.any()
    rv[:, :, 1] = 0
    result = extractor.extract(student(rv, t, y), context=context, source_refs=refs,
                               relative_rotation_current_sensor=r[0].numpy())
    assert result.ray_tokens.shape == (0, 128) and result.ray_ids.size == 0


def test_real_random_frozen_observable_memory_accepts_same_complete_rotation():
    from mtare_topo.representation.primitive_relation_observable_model import ObservableSparsePortRelationNet
    rv, t, y, r, _ = inputs()
    torch.manual_seed(7)
    backbone = ObservableSparsePortRelationNet().eval().requires_grad_(False)
    before = {k: v.clone() for k, v in backbone.state_dict().items()}
    adapter = FrozenSE3EncoderAdapterV1(backbone, nn.Identity())
    result = adapter.extract_compact_features(rv, t, y, relative_rotation_current_sensor=r)
    with torch.no_grad(): memory, mask, xyz, _ = backbone._memory(rv, t, y, relative_rotation_current_sensor=r)
    torch.testing.assert_close(result.frozen_sensor_context, torch.where(mask[..., None], memory, 0.), rtol=0, atol=0)
    means, _ = _token_xyz(result.points_xyz_m.reshape(1, 5, 16, 720, 3), result.valid.reshape(1, 5, 16, 720))
    assert torch.equal(xyz, means.reshape(1, 900, 3))
    assert all(torch.equal(v, backbone.state_dict()[k]) for k, v in before.items())
