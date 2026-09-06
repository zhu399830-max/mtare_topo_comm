"""Only synthetic scans/modules; actual legacy registration and token layout."""
import inspect
from dataclasses import replace

import pytest
import torch
from torch import nn

from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import (
    FrozenDualPathEncoderAdapterV1, expand_features,
)
from mtare_topo.representation.gse_dual_path_model_v1 import DualPathModelV1
from mtare_topo.representation.primitive_relation_model import (
    PrimitiveRelationNet, _token_xyz, register_causal_lidar_points,
)


class SyntheticBackbone(nn.Module):
    def __init__(self, error=None):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.calls = []
        self.error = error

    def _memory(self, range_valid, translation, yaw):
        self.calls.append((self.training, torch.is_grad_enabled(), len(range_valid)))
        points, valid = register_causal_lidar_points(range_valid, translation, yaw)
        xyz, mask = _token_xyz(points, valid)
        b = len(points)
        code = torch.arange(900, device=points.device, dtype=points.dtype)
        memory = (code[None, :, None] + self.scale).expand(b, 900, 128).clone()
        frame = memory.reshape(b, 5, 180, 128).clone()
        xyz, mask = xyz.reshape(b, 900, 3), mask.reshape(b, 900)
        if self.error == "mask":
            mask = mask.clone()
            mask[:, 0] = ~mask[:, 0]
        elif self.error == "xyz":
            xyz = xyz + 1.
        elif self.error == "shape":
            memory = memory[:, :-1]
        elif self.error == "dtype":
            memory = memory.double()
        elif self.error == "valid_nan":
            memory[mask] = float("nan")
        elif self.error == "invalid_nan":
            memory[~mask] = float("nan")
            frame[~mask.reshape(b, 5, 180)] = float("nan")
        elif self.error == "frame_shape":
            frame = frame[:, :-1]
        if self.error == "arity":
            return memory, mask, xyz
        return memory, mask, xyz, frame


class SyntheticStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(131, 2)
        self.last_input = None

    def forward(self, points_xyz_m, frozen_point_context, valid):
        self.last_input = (points_xyz_m, frozen_point_context, valid)
        pooled = torch.cat((points_xyz_m, frozen_point_context), -1).mean(1)
        return self.projection(pooled)


def scans(batch=1):
    raw = torch.zeros(batch, 5, 2, 16, 720)
    for t in range(5):
        raw[:, t, 0, 0, 4] = .2
        raw[:, t, 1, 0, 4] = 1.
        raw[:, t, 0, 15, 5] = .4
        raw[:, t, 1, 15, 5] = 1.
        raw[:, t, 0, 8, 719] = .3
        raw[:, t, 1, 8, 719] = 1.
    translation = torch.tensor([[[1., 2., .5], [.5, 1., .3], [.2, 0., .1], [.1, -.3, .2], [0., 0., 0.]]])
    yaw = torch.tensor([[35., 25., -10., 5., 0.]])
    return raw, translation.expand(batch, -1, -1).clone(), yaw.expand(batch, -1).clone()


def adapter(error=None):
    return FrozenDualPathEncoderAdapterV1(SyntheticBackbone(error), SyntheticStudent())


def test_actual_registration_every_point_and_exact_sensor_token_context():
    a = adapter()
    inputs = scans()
    output = a.extract_features(*inputs)
    points, valid = register_causal_lidar_points(*inputs)
    assert output.points_xyz_m.shape == (1, 57600, 3)
    assert output.frozen_point_context.shape == (1, 57600, 128)
    assert torch.equal(output.points_xyz_m, points.reshape(1, 57600, 3))
    assert torch.equal(output.valid, valid.reshape(1, -1))
    assert output.sensor_token_index.dtype == torch.long
    for t in range(5):
        for elevation, azimuth in ((0, 4), (15, 5), (8, 719)):
            row = t * 16 * 720 + elevation * 720 + azimuth
            expected = t * 180 + azimuth // 4
            assert output.sensor_token_index[0, row].item() == expected
            assert output.frozen_point_context[0, row].eq(expected + .25).all()
    assert output.frozen_point_context[~output.valid].eq(0).all()


def test_actual_legacy_encoder_implementation_compact_context_parity():
    # Random initialization and synthetic scans only, not a checkpoint probe.
    torch.manual_seed(11)
    a = FrozenDualPathEncoderAdapterV1(PrimitiveRelationNet(), SyntheticStudent())
    inputs = scans()
    compact = a.extract_compact_features(*inputs)
    with torch.no_grad():
        memory, mask, _, _ = a.backbone._memory(*inputs)
    expected = torch.where(mask[..., None], memory, 0.)
    assert torch.equal(compact.frozen_sensor_context, expected)
    assert not a.backbone.training
    assert all(not p.requires_grad and p.grad is None for p in a.backbone.parameters())


def test_same_context_does_not_replace_distinct_upper_lower_xyz_with_mean():
    output = adapter().extract_features(*scans())
    lower, upper = 4, 15 * 720 + 5
    assert torch.equal(output.frozen_point_context[0, lower], output.frozen_point_context[0, upper])
    assert not torch.equal(output.points_xyz_m[0, lower], output.points_xyz_m[0, upper])
    assert output.points_xyz_m[0, lower, 2] != output.points_xyz_m[0, upper, 2]


def test_encoder_eval_no_grad_while_student_receives_gradients_and_inputs_unchanged():
    a = adapter()
    original_backbone = {k: v.clone() for k, v in a.backbone.state_dict().items()}
    inputs = scans()
    before = [v.clone() for v in inputs]
    inputs[0].requires_grad_(True)
    a.train()
    assert a.model.training and not a.backbone.training
    a(*inputs).square().mean().backward()
    assert a.backbone.calls == [(False, False, 1)]
    assert all(not p.requires_grad and p.grad is None for p in a.backbone.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.model.parameters())
    assert a.model.projection.weight.grad.abs().sum() > 0
    assert inputs[0].grad is None
    assert all(torch.equal(x, y) for x, y in zip(inputs, before))
    assert all(torch.equal(v, original_backbone[k]) for k, v in a.backbone.state_dict().items())
    assert all(not value.requires_grad for value in a.model.last_input)
    a.eval()
    assert not a.model.training and not a.backbone.training


def test_cache_is_reusable_without_any_backbone_call_or_gradient_graph():
    a = adapter()
    cached = a.extract_features(*scans())
    assert len(a.backbone.calls) == 1
    p = a.model(cached.points_xyz_m, cached.frozen_point_context, cached.valid)
    q = a.model(cached.points_xyz_m, cached.frozen_point_context, cached.valid)
    assert torch.equal(p, q)
    assert len(a.backbone.calls) == 1
    assert all(value.grad_fn is None for value in vars(cached).values())


def test_compact_cache_and_transient_expansion_exact_parity_without_second_encoder():
    a = adapter()
    compact = a.extract_compact_features(*scans())
    assert compact.frozen_sensor_context.shape == (1, 900, 128)
    assert not hasattr(compact, "frozen_point_context")
    expanded = expand_features(compact)
    assert len(a.backbone.calls) == 1
    normal = a.extract_features(*scans())
    assert all(torch.equal(value, getattr(normal, key)) for key, value in vars(expanded).items())
    compact_bytes = sum(value.numel() * value.element_size() for value in vars(compact).values())
    expanded_bytes = sum(value.numel() * value.element_size() for value in vars(expanded).values())
    assert compact_bytes < 2 * 1024**2
    assert expanded_bytes > 25 * 1024**2
    assert compact_bytes * 3024 < 6 * 1024**3


@pytest.mark.parametrize("error", ["index", "memory_shape", "memory_dtype", "nan"])
def test_compact_cache_layout_validation_is_not_a_source_identity_check(error):
    compact = adapter().extract_compact_features(*scans())
    if error == "index":
        compact = replace(compact, sensor_token_index=compact.sensor_token_index.flip(1))
    elif error == "memory_shape":
        compact = replace(compact, frozen_sensor_context=compact.frozen_sensor_context[:, :-1])
    elif error == "memory_dtype":
        compact = replace(compact, frozen_sensor_context=compact.frozen_sensor_context.double())
    else:
        memory = compact.frozen_sensor_context.clone()
        memory[0, 0, 0] = float("nan")
        compact = replace(compact, frozen_sensor_context=memory)
    with pytest.raises(ValueError):
        expand_features(compact)


def test_real_dual_model_accepts_extracted_active_points_and_has_adapter_gradient():
    # Retain all synthetic valid returns; dense invalid padding adds no evidence.
    a = FrozenDualPathEncoderAdapterV1(SyntheticBackbone(), DualPathModelV1("B"))
    cached = a.extract_features(*scans())
    keep = cached.valid[0]
    p = a.model(cached.points_xyz_m[:, keep], cached.frozen_point_context[:, keep], cached.valid[:, keep])
    (p.structure_event_logits.square().mean() + p.primitives.axis_control_m.square().mean() / 2500.).backward()
    gradients = [p.grad for p in a.model.shared_point_adapter.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    assert a.backbone.scale.grad is None


def test_all_empty_observation_skips_encoder_and_retains_empty_mask():
    a = adapter()
    raw, translation, yaw = scans()
    raw.zero_()
    features = a.extract_features(raw, translation, yaw)
    assert not a.backbone.calls
    assert not features.valid.any()
    assert features.points_xyz_m.eq(0).all()
    assert features.frozen_point_context.eq(0).all()
    # No sentinel valid return introduced by the encoder adapter.
    assert features.sensor_token_index.shape == (1, 57600)


def test_mixed_empty_and_supported_rows_preserve_original_batch_order():
    a = adapter()
    raw, translation, yaw = scans(2)
    raw[0].zero_()
    mixed = a.extract_features(raw, translation, yaw)
    single = a.extract_features(raw[1:], translation[1:], yaw[1:])
    assert a.backbone.calls == [(False, False, 1), (False, False, 1)]
    assert not mixed.valid[0].any()
    assert torch.equal(mixed.points_xyz_m[1:], single.points_xyz_m)
    assert torch.equal(mixed.frozen_point_context[1:], single.frozen_point_context)


def test_partially_empty_history_rejected_before_encoder_no_synthetic_fix():
    a = adapter()
    raw, translation, yaw = scans()
    raw[:, 2].zero_()
    with pytest.raises(ValueError, match="partial empty history"):
        a.extract_features(raw, translation, yaw)
    assert not a.backbone.calls


@pytest.mark.parametrize("error", ["mask", "xyz", "shape", "dtype", "valid_nan", "frame_shape", "arity"])
def test_drifted_backbone_contract_rejected(error):
    with pytest.raises(ValueError):
        adapter(error).extract_features(*scans())


def test_invalid_token_nan_never_pollutes_returned_point_context():
    a = adapter("invalid_nan")
    features = a.extract_features(*scans())
    assert torch.isfinite(features.frozen_point_context).all()
    assert features.frozen_point_context[~features.valid].eq(0).all()


@pytest.mark.parametrize("error", ["range", "mask", "motion", "dtype", "shape", "empty_batch", "nan"])
def test_invalid_sensor_contract_rejected_before_memory(error):
    a = adapter()
    raw, translation, yaw = scans()
    if error == "range":
        raw[0, 0, 0, 0, 0] = 1.5
    elif error == "mask":
        raw[0, 0, 1, 0, 0] = .5
    elif error == "motion":
        yaw[0, -1] = 1.
    elif error == "dtype":
        translation = translation.double()
    elif error == "shape":
        raw = raw[:, :-1]
    elif error == "empty_batch":
        raw, translation, yaw = raw[:0], translation[:0], yaw[:0]
    else:
        raw[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        a.extract_features(raw, translation, yaw)
    assert not a.backbone.calls


def test_external_unfreeze_rejected_and_external_train_mode_restored_to_eval():
    a = adapter()
    a.backbone.train()
    a.extract_features(*scans())
    assert a.backbone.calls[-1] == (False, False, 1)
    a.backbone.scale.requires_grad_(True)
    with pytest.raises(ValueError, match="unfrozen"):
        a.extract_features(*scans())


def test_no_teacher_identity_or_checkpoint_io_arguments():
    expected = ["self", "range_valid", "relative_translation_current_sensor_m", "relative_yaw_current_sensor_deg"]
    assert list(inspect.signature(FrozenDualPathEncoderAdapterV1.forward).parameters) == expected
    assert list(inspect.signature(FrozenDualPathEncoderAdapterV1.extract_features).parameters) == expected
    assert list(inspect.signature(FrozenDualPathEncoderAdapterV1.extract_compact_features).parameters) == expected
    with pytest.raises(ValueError):
        FrozenDualPathEncoderAdapterV1(nn.Linear(1, 1), SyntheticStudent())
    backbone = SyntheticBackbone()
    with pytest.raises(ValueError):
        FrozenDualPathEncoderAdapterV1(backbone, backbone)
    student = SyntheticStudent()
    student.shared_parameter = backbone.scale
    with pytest.raises(ValueError, match="share parameters"):
        FrozenDualPathEncoderAdapterV1(backbone, student)
    assert backbone.scale.requires_grad  # validation happens before any freeze
