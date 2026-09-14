"""Candidate point-distinguishing surface-to-axis readout; no teacher inputs.

This is NOT a calibrated detector or graph observation. Nonnegative weights
combine learned surface-to-axis votes, not necessarily surface coordinates.
Membership is learned and may still mix tunnels. Diagnostics expose that
uncertainty; neither a finite output nor a defined chord certifies geometry.
The tensor vote operation is SE(3)-equivariant when offsets rotate as vectors.
The ordinary coordinate MLP/backbone is NOT claimed to be SE(3)-equivariant.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from .primitive_relation_model import (
    MODEL_DIM, MAXIMUM_RANGE_M, HISTORY_FRAMES, ELEVATION_ROWS,
    AZIMUTH_COLUMNS, MEMORY_AZIMUTH_COLUMNS, register_causal_lidar_points,
)


@dataclass(frozen=True)
class PointAxisVote:
    axis_control_m: torch.Tensor
    control_weights: torch.Tensor
    membership_probability: torch.Tensor
    vote_variance_m2: torch.Tensor
    effective_points: torch.Tensor
    membership_support: torch.Tensor
    direction_defined: torch.Tensor


def point_axis_vote(points, offsets, control_logits, membership_logits, valid):
    """Normalize per-point control support after slot/background competition.

    points: [B,N,3], offsets: [B,N,3] or [B,S,N,3], controls: [B,S,3,N].
    Shared offsets are an ablation; slot-conditioned offsets retain ambiguous
    multi-surface returns without forcing them to have one unique target axis.
    The last membership channel is background. Invalid points may have NaN
    padding but valid values must all be finite. Every batch needs evidence.
    No confidence threshold, teacher identity, or uncertainty calibration here.
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must be [B,N,3]")
    b, n, _ = points.shape
    if b == 0 or n == 0:
        raise ValueError("nonempty points required")
    if control_logits.ndim != 4 or control_logits.shape[:1] != (b,) or control_logits.shape[2:] != (3, n):
        raise ValueError("control logits must be [B,S,3,N]")
    s = control_logits.shape[1]
    if offsets.shape not in (points.shape, (b, s, n, 3)):
        raise ValueError("offsets must be [B,N,3] or [B,S,N,3]")
    if s < 1 or membership_logits.shape != (b, s + 1, n):
        raise ValueError("membership must include S slots and one background")
    if valid.shape != (b, n) or valid.dtype != torch.bool or valid.device != points.device:
        raise ValueError("valid must be bool [B,N] on the points device")
    values = (points, offsets, control_logits, membership_logits)
    if any(v.dtype not in (torch.float32, torch.float64) or v.dtype != points.dtype
           or v.device != points.device for v in values):
        raise ValueError("vote inputs need common float32/float64 dtype/device")
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("all-invalid observation cannot produce geometry")
    offset_mask = valid[..., None] if offsets.ndim == 3 else valid[:, None, :, None]
    masks = (valid[..., None], offset_mask, valid[:, None, None], valid[:, None])
    cleaned = []
    for value, mask in zip(values, masks):
        clean = torch.where(mask, value, torch.zeros_like(value))
        if not bool(torch.isfinite(clean).all()):
            raise ValueError("nonfinite value at valid point")
        cleaned.append(clean)
    points, offsets, controls, members = cleaned
    log_members = torch.log_softmax(members, dim=1)
    log_weights = controls + log_members[:, :s, None]
    weights = torch.softmax(log_weights.masked_fill(~valid[:, None, None], -torch.inf), dim=-1)
    probability = torch.exp(log_members).masked_fill(~valid[:, None], 0.)
    # Center moments for stability under large common coordinate translations.
    first = valid.to(torch.int64).argmax(dim=-1)
    center = points[torch.arange(b, device=points.device), first]
    if offsets.ndim == 3:
        offsets = offsets[:, None]
    votes = torch.where(valid[:, None, :, None], points[:, None] - center[:, None, None] + offsets, 0.)
    # Shared-offset ablation has one vote set, broadcast over all S slots.
    votes = votes.expand(-1, s, -1, -1)
    relative_axis = torch.einsum("bscn,bsnd->bscd", weights, votes)
    axis = relative_axis + center[:, None, None]
    second = torch.einsum("bscn,bsn->bsc", weights, votes.square().sum(dim=-1))
    variance = (second - relative_axis.square().sum(dim=-1)).clamp_min(0.)
    support = torch.einsum("bscn,bsn->bsc", weights, probability[:, :s])
    effective = weights.square().sum(dim=-1).reciprocal()
    tolerance = 32 * torch.finfo(points.dtype).eps * axis.abs().amax(dim=(-1, -2)).clamp_min(1.)
    defined = torch.linalg.vector_norm(axis[:, :, 2] - axis[:, :, 0], dim=-1) > tolerance
    if not all(bool(torch.isfinite(v).all()) for v in (axis, variance, support, effective)):
        raise ValueError("nonfinite vote output")
    return PointAxisVote(axis, weights, probability, variance, effective, support, defined)


@dataclass(frozen=True)
class PointAxisPrediction:
    votes: PointAxisVote
    surface_to_axis_offset_m: torch.Tensor
    membership_logits: torch.Tensor


class PointAxisReadout(nn.Module):
    """Thin geometry branch over existing fused memory and primitive queries.

    Point context uses its parent SENSOR token, not its teacher primitive.
    Coordinates distinguish returns sharing a compressed token. Point offsets
    have a low-rank slot-conditioned component, so ambiguous multi-surface
    returns need not vote for one universal axis. Each query/control learns
    its own point support. Three controls remain
    unconstrained: no new straightness/monotonicity target is imposed.
    """
    def __init__(self, model_dim=MODEL_DIM, point_dim=32):
        super().__init__()
        if type(model_dim) is not int or type(point_dim) is not int or min(model_dim, point_dim) < 1:
            raise ValueError("positive integer feature dimensions required")
        self.model_dim, self.point_dim = model_dim, point_dim
        self.context = nn.Linear(model_dim, point_dim)
        self.geometry = nn.Sequential(nn.Linear(6, point_dim), nn.SiLU(), nn.Linear(point_dim, point_dim))
        self.key = nn.Linear(point_dim, point_dim, bias=False)
        self.membership_query = nn.Linear(model_dim, point_dim)
        self.control_query = nn.Linear(model_dim, 3 * point_dim)
        self.background = nn.Linear(point_dim, 1)
        self.offset = nn.Linear(point_dim, 3)
        self.slot_offset = nn.Linear(model_dim, point_dim * 3)
        # Starts as a point-distinguishing RAW-coordinate baseline, not the old
        # mean-coordinate model. Zero output layer still has nonzero gradients.
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.slot_offset.weight)
        nn.init.zeros_(self.slot_offset.bias)

    def forward(self, points, valid, memory, memory_xyz, sensor_token_index, slots):
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must be [B,N,3]")
        b, n, _ = points.shape
        if memory.ndim != 3 or memory.shape[0] != b or memory.shape[-1] != self.model_dim:
            raise ValueError("memory feature shape drift")
        if memory_xyz.shape != memory.shape[:2] + (3,) or slots.ndim != 3 or slots.shape[0] != b or slots.shape[-1] != self.model_dim:
            raise ValueError("memory coordinates or slot shape drift")
        if valid.shape != (b, n) or valid.dtype != torch.bool:
            raise ValueError("point validity shape/dtype drift")
        if sensor_token_index.shape != (b, n) or sensor_token_index.dtype != torch.long:
            raise ValueError("sensor token index must be int64 [B,N]")
        tensors = (valid, memory, memory_xyz, sensor_token_index, slots)
        if any(v.device != points.device for v in tensors):
            raise ValueError("readout devices must match")
        if any(v.dtype != points.dtype for v in (memory, memory_xyz, slots)):
            raise ValueError("readout floating dtypes must match")
        if points.dtype not in (torch.float32, torch.float64):
            raise ValueError("readout requires float32 or float64")
        if bool(((sensor_token_index < 0) | (sensor_token_index >= memory.shape[1])).any()):
            raise ValueError("sensor token index out of bounds")
        clean = torch.where(valid[..., None], points, 0.)
        if not all(bool(torch.isfinite(v).all()) for v in (clean, memory, memory_xyz, slots)):
            raise ValueError("nonfinite readout inputs")
        index = sensor_token_index[..., None]
        context = self.context(memory).gather(1, index.expand(-1, -1, self.point_dim))
        anchor = memory_xyz.gather(1, index.expand(-1, -1, 3))
        geometry = torch.cat((clean, clean - anchor), dim=-1) / MAXIMUM_RANGE_M
        feature = torch.nn.functional.silu(context + self.geometry(geometry))
        feature = torch.where(valid[..., None], feature, 0.)
        keys = self.key(feature)
        membership = torch.einsum("bsd,bnd->bsn", self.membership_query(slots), keys) / math.sqrt(self.point_dim)
        membership = torch.cat((membership, self.background(feature).transpose(1, 2)), dim=1)
        query = self.control_query(slots).reshape(b, slots.shape[1], 3, self.point_dim)
        controls = torch.einsum("bscd,bnd->bscn", query, keys) / math.sqrt(self.point_dim)
        slot_matrix = self.slot_offset(slots).reshape(b, slots.shape[1], self.point_dim, 3)
        offsets = (self.offset(feature)[:, None] + torch.einsum("bnd,bsdc->bsnc", feature, slot_matrix)
                   / math.sqrt(self.point_dim)) * MAXIMUM_RANGE_M
        offsets = torch.where(valid[:, None, :, None], offsets, 0.)
        votes = point_axis_vote(clean, offsets, controls, membership, valid)
        return PointAxisPrediction(votes, offsets, membership)


class FrozenBackbonePointAxisAdapter(nn.Module):
    """No checkpoint I/O: caller supplies an existing, already verified model.

    Only range/valid and causal relative motion enter deployment forward.
    Old relation logits are deliberately NOT reused with changed geometry.
    Freezing mutates the supplied module's training/gradient flags explicitly;
    it never changes parameter values. Keep a separate old model for ablation.
    """
    def __init__(self, backbone, readout=None):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.readout = PointAxisReadout() if readout is None else readout

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, range_valid, relative_translation_current_sensor_m, relative_yaw_current_sensor_deg):
        with torch.no_grad():
            points, valid = register_causal_lidar_points(range_valid,
                relative_translation_current_sensor_m, relative_yaw_current_sensor_deg)
            memory, memory_valid, memory_xyz, _ = self.backbone._memory(range_valid,
                relative_translation_current_sensor_m, relative_yaw_current_sensor_deg)
            queries = self.backbone.slot_query[None].expand(len(points), -1, -1)
            slots = self.backbone.slot_decoder(queries, memory, memory_key_padding_mask=~memory_valid)
        # Fixed scan tensor layout gives SENSOR-token provenance, not GT grouping.
        indices = (torch.arange(HISTORY_FRAMES, device=points.device)[:, None, None] * MEMORY_AZIMUTH_COLUMNS
                   + torch.arange(AZIMUTH_COLUMNS, device=points.device)[None, None] // 4)
        indices = indices.expand(HISTORY_FRAMES, ELEVATION_ROWS, AZIMUTH_COLUMNS)
        indices = indices.reshape(1, -1).expand(len(points), -1)
        return self.readout(points.reshape(len(points), -1, 3), valid.reshape(len(points), -1),
            memory, memory_xyz, indices, slots)
