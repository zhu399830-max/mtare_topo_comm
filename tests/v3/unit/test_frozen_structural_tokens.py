"""Synthetic modules/scans only; actual shared registration, patches and C path."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from mtare_topo.integration.frozen_structural_tokens import FrozenStructuralTokenExtractor
from mtare_topo.representation.branch_relation_learning import RayBranchRelationModel
from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import FrozenDualPathEncoderAdapterV1
from mtare_topo.representation.gse_structure_context import CausalFrameOrderContext
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points, _token_xyz


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.25))
        self.calls = []

    def _memory(self, *args):
        self.calls.append((self.training, torch.is_grad_enabled()))
        points, valid = register_causal_lidar_points(*args)
        xyz, mask = _token_xyz(points, valid)
        memory = (torch.arange(900, device=points.device, dtype=points.dtype)[None, :, None]
                  + self.scale).expand(1, 900, 128).clone()
        return memory, mask.reshape(1, 900), xyz.reshape(1, 900, 3), memory.reshape(1, 5, 180, 128)


@pytest.fixture
def setup():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(14)
    encoder = FakeBackbone()
    model = RayBranchRelationModel('C')
    adapter = FrozenDualPathEncoderAdapterV1(encoder, nn.Identity())
    extractor = FrozenStructuralTokenExtractor(adapter, model)
    ranges = np.zeros((5, 16, 720), dtype=np.float32)
    valid = np.zeros(ranges.shape, dtype=np.uint8)
    for frame in range(5):
        for row, col, distance in [(0, 4, 4.), (15, 5, 4.), (8, 700, 20.)]:
            ranges[frame, row, col] = distance
            valid[frame, row, col] = 1
    translation = np.zeros((5, 3), dtype=np.float32)
    translation[:4, 2] = [.4, .3, .2, .1]
    student = SimpleNamespace(ranges_m=ranges, valid_mask=valid,
        relative_translation_current_sensor_m=translation,
        relative_yaw_current_sensor_deg=np.array([10, 5, -5, 2, 0], dtype=np.float32))
    context = CausalFrameOrderContext('sensor_current', (4, 9, 12, 20, 24), 24, 'synthetic-bag')
    refs = tuple(f'bag://original/stamp-{i}' for i in (4, 9, 12, 20, 24))
    yield SimpleNamespace(extractor=extractor, model=model, encoder=encoder,
                          student=student, context=context, refs=refs)
    torch.set_num_threads(previous)


def extract(s, **kwargs):
    return s.extractor.extract(s.student, context=s.context, source_refs=s.refs, **kwargs)


def test_actual_common_path_keeps_each_ray_xyz_context_support_and_source(setup):
    s = setup
    state = {k: v.clone() for k, v in s.model.state_dict().items()}
    seen = []
    handle = s.model.register_forward_pre_hook(lambda m, args: seen.append(args))
    output = extract(s)
    handle.remove()
    assert s.encoder.calls == [(False, False)]
    assert seen[0][1].shape == (0, 2)
    assert torch.equal(seen[0][0].observed_features[:, 0],
                       torch.tensor([721.25, 895.25, 721.25]))
    np.testing.assert_array_equal(output.ray_ids, [4, 8 * 720 + 700, 15 * 720 + 5])
    np.testing.assert_allclose(np.linalg.norm(output.endpoints_current_sensor_m, axis=1), [4, 20, 4], atol=1e-5)
    assert output.endpoints_current_sensor_m[0, 2] < output.endpoints_current_sensor_m[2, 2]
    assert output.ray_tokens.shape == (3, 128)
    assert not np.array_equal(output.ray_tokens[0], output.ray_tokens[2])
    assert output.geometric_support.all()  # nearest patches, not ray membership
    assert output.nearest_patch_indices.shape == (3, min(8, len(output.patch_centers_current_sensor_m)))
    assert output.patch_frame_support.shape == (len(output.patch_centers_current_sensor_m), 5)
    assert output.source_refs == s.refs and output.context is s.context
    assert not output.support_is_membership_probability
    assert not output.ray_tokens.flags.writeable
    for field in ('ray_ids', 'endpoints_current_sensor_m', 'ray_tokens',
                  'geometric_support', 'nearest_patch_indices',
                  'patch_centers_current_sensor_m', 'patch_frame_support', 'sensor_token_indices'):
        with pytest.raises(ValueError):
            getattr(output, field).setflags(write=True)
    assert all(torch.equal(v, state[k]) for k, v in s.model.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in s.model.parameters())
    np.testing.assert_array_equal(output.ray_tokens, extract(s).ray_tokens)


def test_no_patches_retains_distant_rays_without_geometric_support(setup):
    s = setup
    s.student.ranges_m[s.student.valid_mask.astype(bool)] = 20
    s.student.relative_translation_current_sensor_m[:] = 0
    output = extract(s)
    assert output.ray_tokens.shape == (3, 128)
    assert output.patch_centers_current_sensor_m.shape == (0, 3)
    assert output.nearest_patch_indices.shape == (3, 0)
    assert not output.geometric_support.any()
    assert np.isfinite(output.ray_tokens).all()


def test_all_empty_is_empty_and_partial_empty_history_rejected(setup):
    s = setup
    s.student.valid_mask[:] = 0
    output = extract(s)
    assert output.ray_tokens.shape == (0, 128)
    assert output.ray_ids.size == 0 and s.encoder.calls == []
    s.student.valid_mask[4, 0, 4] = 1
    with pytest.raises(ValueError, match='partial empty history'):
        extract(s)


@pytest.mark.parametrize('problem', ['mask', 'range', 'nan', 'motion', 'dtype', 'context', 'source', 'unfreeze'])
def test_invalid_input_fails_before_encoder(setup, problem):
    s = setup
    if problem == 'mask': s.student.valid_mask[0, 0, 4] = 2
    elif problem == 'range': s.student.ranges_m[0, 0, 4] = 0
    elif problem == 'nan': s.student.ranges_m[0, 0, 4] = np.nan
    elif problem == 'motion': s.student.relative_translation_current_sensor_m[-1, 0] = 1
    elif problem == 'dtype': s.student.ranges_m = s.student.ranges_m.astype(np.float64)
    elif problem == 'context': s.context = replace(s.context, observation_frame_id=25)
    elif problem == 'source': s.refs = ('',) * 5
    elif problem == 'unfreeze': s.model.requires_grad_(True)
    with pytest.raises(ValueError): extract(s)
    assert s.encoder.calls == []


def test_full_rotation_contract_rejects_silent_roll_pitch_loss(setup):
    s = setup
    angles = np.deg2rad(s.student.relative_yaw_current_sensor_deg.astype(np.float64))
    rot = np.zeros((5, 3, 3), dtype=np.float32)
    rot[:, 0, 0] = rot[:, 1, 1] = np.cos(angles)
    rot[:, 1, 0] = np.sin(angles)
    rot[:, 0, 1] = -np.sin(angles)
    rot[:, 2, 2] = 1
    np.testing.assert_array_equal(extract(s).ray_tokens,
                                  extract(s, relative_rotation_current_sensor=rot).ray_tokens)
    rot[0, 2, 0] = .01
    with pytest.raises(ValueError, match='yaw-only'):
        extract(s, relative_rotation_current_sensor=rot)


def test_provenance_changes_are_not_model_features(setup):
    s = setup
    old = extract(s)
    s.context = CausalFrameOrderContext('sensor_current', (100, 110, 120, 130, 140), 140, 'other-source')
    s.refs = tuple('new-' + ref for ref in s.refs)
    new = extract(s)
    np.testing.assert_array_equal(old.ray_tokens, new.ray_tokens)
    assert old.source_refs != new.source_refs
