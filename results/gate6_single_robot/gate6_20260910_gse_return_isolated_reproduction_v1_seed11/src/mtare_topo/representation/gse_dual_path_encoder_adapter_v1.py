"""Frozen legacy encoder -> raw-point dual-path features, with no file I/O.

The caller supplies already loaded modules. Constructing this adapter freezes
the supplied backbone's parameters and mode, not its values; use a separate
instance if that backbone is needed for training elsewhere. A/B/C can consume
the same compact extracted feature object. No teacher/ID filters or mean XYZ
substitute. Cache 900 sensor features, NOT 57600 replicated point features:
expanded features are transient single-batch tensors, never a dataset cache.

Sensor-token indices express tensor-layout provenance, NOT physical identity.
The external producer must bind source frames, coordinate frame, source/weight
hashes and cache rows. Matching validity/coordinates cannot authenticate them.
This adapter does not repair or waive the model's unresolved yaw contract.
"""
from dataclasses import dataclass

import torch
from torch import nn

from .primitive_relation_model import (
    AZIMUTH_COLUMNS, ELEVATION_ROWS, HISTORY_FRAMES, MEMORY_AZIMUTH_COLUMNS,
    MODEL_DIM, _token_xyz, register_causal_lidar_points,
)


@dataclass(frozen=True)
class FrozenDualPathFeatures:
    """Transient expanded model input; do not persist population-sized copies."""
    points_xyz_m: torch.Tensor          # B,57600,3; current sensor coordinates
    frozen_point_context: torch.Tensor  # B,57600,128; invalid returns zero
    valid: torch.Tensor                 # B,57600 bool
    sensor_token_index: torch.Tensor    # B,57600 int64; t*180 + azimuth//4


@dataclass(frozen=True)
class CompactFrozenDualPathFeatures:
    """Cacheable numerical payload; external producer must bind source identity."""
    points_xyz_m: torch.Tensor          # B,57600,3, not token means
    frozen_sensor_context: torch.Tensor # B,900,128, invalid tokens zero
    valid: torch.Tensor                 # B,57600 bool
    sensor_token_index: torch.Tensor    # B,57600 int64 (reconstructible layout)


def expand_features(compact):
    """Gather one batch's compact cache; this does not authenticate its source."""
    if not isinstance(compact, CompactFrozenDualPathFeatures):
        raise ValueError("compact feature type required")
    points, memory, valid, index = (compact.points_xyz_m, compact.frozen_sensor_context,
                                   compact.valid, compact.sensor_token_index)
    n = HISTORY_FRAMES * ELEVATION_ROWS * AZIMUTH_COLUMNS
    b = points.shape[0] if torch.is_tensor(points) and points.ndim == 3 else 0
    if (b < 1 or points.shape != (b, n, 3) or not all(torch.is_tensor(v) for v in (memory, valid, index)) or
            memory.shape != (b, HISTORY_FRAMES * MEMORY_AZIMUTH_COLUMNS, MODEL_DIM) or
            valid.shape != (b, n) or index.shape != (b, n)):
        raise ValueError("compact feature shape drift")
    if (points.dtype not in (torch.float32, torch.float64) or memory.dtype != points.dtype or
            valid.dtype != torch.bool or index.dtype != torch.long or
            any(v.device != points.device for v in (memory, valid, index))):
        raise ValueError("compact feature dtype/device drift")
    if not bool(torch.isfinite(points).all() and torch.isfinite(memory).all()):
        raise ValueError("nonfinite compact feature")
    t = torch.arange(HISTORY_FRAMES, device=points.device)[:, None, None]
    azimuth = torch.arange(AZIMUTH_COLUMNS, device=points.device)[None, None]
    expected = (t * MEMORY_AZIMUTH_COLUMNS + azimuth // 4).expand(
        HISTORY_FRAMES, ELEVATION_ROWS, AZIMUTH_COLUMNS).reshape(1, n).expand(b, -1)
    if not torch.equal(index, expected):
        raise ValueError("compact sensor-token layout drift")
    context = memory.detach().gather(1, index[..., None].expand(-1, -1, MODEL_DIM))
    context = torch.where(valid[..., None], context, 0.)
    return FrozenDualPathFeatures(points.detach(), context, valid.detach(), index.detach())


class FrozenDualPathEncoderAdapterV1(nn.Module):
    def __init__(self, backbone, model):
        super().__init__()
        if not isinstance(backbone, nn.Module) or not callable(getattr(backbone, "_memory", None)):
            raise ValueError("backbone must be a supplied module with _memory")
        if not isinstance(model, nn.Module) or model is backbone:
            raise ValueError("a separate trainable observation model is required")
        if {id(p) for p in backbone.parameters()} & {id(p) for p in model.parameters()}:
            raise ValueError("backbone and observation model must not share parameters")
        self.backbone = backbone
        self.model = model
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def extract_compact_features(self, range_valid, relative_translation_current_sensor_m,
                                 relative_yaw_current_sensor_deg):
        """Pure frozen inference; compact tensors can be cached by the caller.

        Entirely empty observations bypass the legacy encoder and remain empty.
        A partially empty five-frame history is rejected: legacy _memory cannot
        encode it without changing its contract. No synthetic valid ray is added.
        Source binding and serialization are deliberately outside this method.
        """
        self.backbone.eval()
        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            raise ValueError("backbone was unfrozen outside the adapter")
        inputs = (range_valid, relative_translation_current_sensor_m, relative_yaw_current_sensor_deg)
        if (not all(torch.is_tensor(value) for value in inputs) or
                range_valid.dtype not in (torch.float32, torch.float64) or
                any(value.dtype != range_valid.dtype or value.device != range_valid.device for value in inputs)):
            raise ValueError("observation inputs need a common float32/64 dtype/device")
        if range_valid.ndim == 0 or range_valid.shape[0] < 1:
            raise ValueError("empty batch is not an observation")
        points, valid = register_causal_lidar_points(*inputs)
        b = len(points)
        expected_xyz, token_valid = _token_xyz(points, valid)
        frame_supported = token_valid.any(-1)
        observation_supported = frame_supported.any(-1)
        if bool((observation_supported & ~frame_supported.all(-1)).any()):
            raise ValueError("legacy encoder requires all five frames supported; partial empty history")
        flat_valid = valid.reshape(b, -1)
        n = flat_valid.shape[1]
        t = torch.arange(HISTORY_FRAMES, device=points.device)[:, None, None]
        azimuth = torch.arange(AZIMUTH_COLUMNS, device=points.device)[None, None]
        index = (t * MEMORY_AZIMUTH_COLUMNS + azimuth // 4)
        index = index.expand(HISTORY_FRAMES, ELEVATION_ROWS, AZIMUTH_COLUMNS)
        index = index.reshape(1, n).expand(b, -1).clone()
        context = points.new_zeros((b, HISTORY_FRAMES * MEMORY_AZIMUTH_COLUMNS, MODEL_DIM))
        if bool(observation_supported.any()):
            keep = observation_supported
            output = self.backbone._memory(*(value[keep] for value in inputs))
            if not isinstance(output, (tuple, list)) or len(output) != 4:
                raise ValueError("legacy _memory must return its four-field contract")
            memory, memory_valid, memory_xyz, frame_memory = output
            k = int(keep.sum())
            shapes = ((k, HISTORY_FRAMES * MEMORY_AZIMUTH_COLUMNS, MODEL_DIM),
                      (k, HISTORY_FRAMES * MEMORY_AZIMUTH_COLUMNS),
                      (k, HISTORY_FRAMES * MEMORY_AZIMUTH_COLUMNS, 3),
                      (k, HISTORY_FRAMES, MEMORY_AZIMUTH_COLUMNS, MODEL_DIM))
            if any(not torch.is_tensor(value) or value.shape != shape
                   for value, shape in zip(output, shapes)):
                raise ValueError("legacy memory shape drift")
            if (memory_valid.dtype != torch.bool or
                    any(value.device != points.device for value in output) or
                    any(value.dtype != points.dtype for value in (memory, memory_xyz, frame_memory))):
                raise ValueError("legacy memory dtype/device drift")
            expected_mask = token_valid[keep].reshape(k, -1)
            if not torch.equal(memory_valid, expected_mask):
                raise ValueError("legacy token validity disagrees with registered returns")
            if not torch.equal(memory_xyz, expected_xyz[keep].reshape(k, -1, 3)):
                raise ValueError("legacy token coordinates disagree with registered returns")
            cleaned_memory = torch.where(memory_valid[..., None], memory, 0.)
            cleaned_frame = torch.where(token_valid[keep][..., None], frame_memory, 0.)
            if not all(bool(torch.isfinite(value).all()) for value in (cleaned_memory, cleaned_frame)):
                raise ValueError("nonfinite supported legacy context")
            context[keep] = cleaned_memory
        return CompactFrozenDualPathFeatures(points.reshape(b, n, 3).detach(), context.detach(),
                                             flat_valid.detach(), index.detach())

    def extract_features(self, range_valid, relative_translation_current_sensor_m,
                         relative_yaw_current_sensor_deg):
        """Transient expanded convenience interface; use compact extraction for cache."""
        return expand_features(self.extract_compact_features(range_valid,
            relative_translation_current_sensor_m, relative_yaw_current_sensor_deg))

    def forward(self, range_valid, relative_translation_current_sensor_m,
                relative_yaw_current_sensor_deg):
        features = self.extract_features(range_valid, relative_translation_current_sensor_m,
                                         relative_yaw_current_sensor_deg)
        return self.model(features.points_xyz_m, features.frozen_point_context, features.valid)
