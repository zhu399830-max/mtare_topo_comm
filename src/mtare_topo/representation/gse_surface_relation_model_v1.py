"""Paired A raw / B raw+surface unary / C raw+surface relations prototype.

All branches inherit any supplied encoder pretraining. A means no explicit
patch readout, NOT no geometric pretraining. No encoder/teacher/identity file
is loaded. B/C have identical parameter layouts: B's three message layers
are unary self updates, C additionally receives sparse neighbor messages.
Neither this coordinate MLP nor fixed-voxel extraction is SE(3) equivariant.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .gse_surface_patches_v1 import SurfacePatches, build_patch_neighbors

DIM = 128
ANCHORS = 32
OPENINGS = 64
REACHABILITY_ORDER = ("traversable", "blocked", "unknown")


@dataclass(frozen=True)
class SurfacePatchBatch:
    unary: torch.Tensor                  # B,M,18
    valid: torch.Tensor                  # B,M; degenerate normals remain present
    neighbor_index: torch.Tensor         # B,M,8; -1 padding
    neighbor_valid: torch.Tensor
    relation: torch.Tensor               # B,M,8,9; relativeXYZ/dist/normal/valid/ray3


def collate_surface_patches(patches, *, device="cpu", dtype=torch.float32):
    """Geometry-only batching; no GT filtering and no dropped occupied cells."""
    if not isinstance(patches, (list, tuple)) or not patches or any(type(p) is not SurfacePatches for p in patches):
        raise ValueError("nonempty batch of extracted SurfacePatches required")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("floating patch tensors required")
    b, m = len(patches), max(1, max(len(p.centers_m) for p in patches))
    if m > 4096:
        raise OverflowError("patch capacity4096 exceeded")
    unary = np.zeros((b, m, 18)); valid = np.zeros((b, m), dtype=bool)
    index = np.full((b, m, 8), -1, dtype=np.int64); neighbor_valid = np.zeros((b, m, 8), dtype=bool)
    relation = np.zeros((b, m, 8, 9))
    for i, p in enumerate(patches):
        n = len(p.centers_m); r = build_patch_neighbors(p)
        valid[i, :n] = True
        unary[i, :n] = np.concatenate((p.centers_m / p.roi_radius_m, p.normals,
            p.normal_valid[:, None].astype(float), (p.bounds_max_m - p.bounds_min_m) / p.voxel_size_m,
            p.roughness_m[:, None] / p.voxel_size_m, np.log1p(p.point_count[:, None]) / math.log(57601),
            (p.frame_support > 0).astype(float), p.normal_uncertainty[:, None]), axis=-1)
        index[i, :n] = r.neighbor_index; neighbor_valid[i, :n] = r.valid
        relation[i, :n] = np.concatenate((r.relative_xyz_m / p.roi_radius_m,
            r.distance_m[..., None] / p.roi_radius_m, r.normal_abs_dot[..., None],
            r.normal_pair_valid[..., None].astype(float), r.ray_evidence), axis=-1)
    return SurfacePatchBatch(torch.tensor(unary, device=device, dtype=dtype),
        torch.tensor(valid, device=device), torch.tensor(index, device=device),
        torch.tensor(neighbor_valid, device=device), torch.tensor(relation, device=device, dtype=dtype))


@dataclass(frozen=True)
class SurfaceRelationPrediction:
    anchor_position_m: torch.Tensor
    anchor_presence_logits: torch.Tensor
    anchor_uncertainty_m: torch.Tensor
    opening_position_m: torch.Tensor
    opening_presence_logits: torch.Tensor
    opening_direction: torch.Tensor
    opening_direction_valid: torch.Tensor
    opening_dimensions_m: torch.Tensor
    opening_dimension_evidence_logits: torch.Tensor
    opening_support_logits: torch.Tensor
    reachability_logits: torch.Tensor
    membership_logits: torch.Tensor
    membership_validity_logits: torch.Tensor
    observation_supported: torch.Tensor


class _MessageLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(2 * DIM + 9, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.update = nn.Sequential(nn.Linear(2 * DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.norm = nn.LayerNorm(DIM)

    def forward(self, x, patch, use_relations):
        if use_relations:
            b, m, d = x.shape; safe = patch.neighbor_index.clamp_min(0)
            neighbor = x[torch.arange(b, device=x.device)[:, None, None], safe]
            sender = x[:, :, None].expand(-1, -1, 8, -1)
            messages = self.message(torch.cat((sender, neighbor, patch.relation.detach()), -1))
            mask = patch.neighbor_valid & patch.valid[..., None]
            aggregate = (messages * mask[..., None]).sum(2) / mask.sum(2).clamp_min(1)[..., None]
            # Isolated nodes still receive a unary update; no fake graph edge.
            self_message = self.message(torch.cat((x, x, x.new_zeros(b, m, 9)), -1))
            aggregate = torch.where(mask.any(2)[..., None], aggregate, self_message)
        else:
            aggregate = self.message(torch.cat((x, x, x.new_zeros(*x.shape[:2], 9)), -1))
        out = self.norm(x + self.update(torch.cat((x, aggregate), -1)))
        return torch.where(patch.valid[..., None], out, 0.)


class SurfaceRelationModelV1(nn.Module):
    """Teacher-free fixed32+64 query decoder, sparse three-layer patch path.

    Inputs use the same current sensor frame. For large raw point clouds supply
    the compact adapter's sensor_token_index: nonlinear point features are
    pooled into900 layout-provenance bins. No world/node/edge IDs are accepted.
    Small synthetic clouds <=900 can use one token per point. Population-wide
    expanded128-D point caching is deliberately outside this API.
    """
    def __init__(self, path):
        super().__init__()
        if path not in ("A", "B", "C"):
            raise ValueError("path must be A/B/C")
        self.path = path
        self.raw_adapter = nn.Sequential(nn.Linear(3 + DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM), nn.LayerNorm(DIM))
        self.patch_adapter = nn.Sequential(nn.Linear(18, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.patch_layers = nn.ModuleList(_MessageLayer() for _ in range(3))
        self.anchor_queries = nn.Parameter(torch.randn(ANCHORS, DIM) / math.sqrt(DIM))
        self.opening_queries = nn.Parameter(torch.randn(OPENINGS, DIM) / math.sqrt(DIM))
        layer = nn.TransformerDecoderLayer(DIM, 4, 4 * DIM, dropout=0., activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, 3, norm=nn.LayerNorm(DIM))
        self.anchor_position = nn.Linear(DIM, 3); self.anchor_presence = nn.Linear(DIM, 1)
        self.anchor_uncertainty = nn.Linear(DIM, 3)
        self.opening_position = nn.Linear(DIM, 3); self.opening_presence = nn.Linear(DIM, 1)
        self.opening_orientation = nn.Linear(DIM, 3); self.opening_dimensions = nn.Linear(DIM, 2)
        self.dimension_evidence = nn.Linear(DIM, 2); self.opening_support = nn.Linear(DIM, 1)
        self.reachability = nn.Linear(DIM, 3)
        self.member_anchor = nn.Linear(DIM, DIM); self.member_opening = nn.Linear(DIM, DIM)
        self.validity_anchor = nn.Linear(DIM, DIM); self.validity_opening = nn.Linear(DIM, DIM)

    def _raw(self, points, context, valid, sensor_token_index):
        if (not torch.is_tensor(points) or points.ndim != 3 or points.shape[-1] != 3 or points.shape[0] < 1
                or points.shape[1] < 1 or points.dtype not in (torch.float32, torch.float64)
                or not torch.is_tensor(context) or context.shape != (*points.shape[:2], DIM)
                or context.dtype != points.dtype or context.device != points.device
                or not torch.is_tensor(valid) or valid.shape != points.shape[:2] or valid.dtype != torch.bool or valid.device != points.device):
            raise ValueError("same-frame B,N,3 XYZ, B,N,128 context and bool validity required")
        xyz = torch.where(valid[..., None], points, 0.)
        ctx = torch.where(valid[..., None], context.detach(), 0.)
        if not bool(torch.isfinite(xyz).all() and torch.isfinite(ctx).all()):
            raise ValueError("nonfinite observed point/context")
        features = self.raw_adapter(torch.cat((xyz / 10., ctx), -1))
        features = torch.where(valid[..., None], features, 0.)
        if sensor_token_index is None:
            if points.shape[1] > 900:
                raise ValueError("large clouds require explicit compact sensor-token layout")
            return features, valid
        index = sensor_token_index
        if (not torch.is_tensor(index) or index.shape != valid.shape or index.dtype != torch.long
                or index.device != points.device or bool(((index < 0) | (index >= 900)).any())):
            raise ValueError("bounded sensor-token layout0..899 required, not physical IDs")
        b = len(points); pooled = features.new_zeros(b, 900, DIM)
        counts = features.new_zeros(b, 900)
        pooled.scatter_add_(1, index[..., None].expand(-1, -1, DIM), features)
        counts.scatter_add_(1, index, valid.to(features.dtype))
        return pooled / counts.clamp_min(1)[..., None], counts > 0

    def _patch(self, patch, b, dtype, device):
        if not isinstance(patch, SurfacePatchBatch):
            raise ValueError("explicit surface patch batch required for B/C")
        x = patch.unary; m = x.shape[1] if x.ndim == 3 else 0
        if x.shape != (b, m, 18) or not 1 <= m <= 4096 or x.dtype != dtype or x.device != device:
            raise ValueError("patch unary shape/capacity/dtype drift")
        if (patch.valid.shape != (b, m) or patch.valid.dtype != torch.bool
                or patch.neighbor_index.shape != (b, m, 8) or patch.neighbor_index.dtype != torch.long
                or patch.neighbor_valid.shape != (b, m, 8) or patch.neighbor_valid.dtype != torch.bool
                or patch.relation.shape != (b, m, 8, 9) or patch.relation.dtype != dtype
                or any(t.device != device for t in (patch.valid, patch.neighbor_index, patch.neighbor_valid, patch.relation))):
            raise ValueError("patch sparse relation shape/dtype drift")
        if bool(((patch.neighbor_index < -1) | (patch.neighbor_index >= m)).any()) or not torch.equal(patch.neighbor_valid, patch.neighbor_index >= 0):
            raise ValueError("neighbor indices/masks disagree")
        safe = patch.neighbor_index.clamp_min(0)
        target_valid = patch.valid[torch.arange(b, device=device)[:, None, None], safe]
        if bool((patch.neighbor_valid & (~target_valid | ~patch.valid[..., None])).any()):
            raise ValueError("neighbor refers to padded patch")
        unary = torch.where(patch.valid[..., None], x.detach(), 0.)
        if not bool(torch.isfinite(unary).all() and torch.isfinite(patch.relation).all()):
            raise ValueError("nonfinite patch attributes/relations")
        feature = self.patch_adapter(unary)
        for layer in self.patch_layers:
            feature = layer(feature, patch, self.path == "C")
        return feature, patch.valid

    def forward(self, points_xyz_m, frozen_point_context, valid, patches=None, *, sensor_token_index=None):
        memory, memory_valid = self._raw(points_xyz_m, frozen_point_context, valid, sensor_token_index)
        supported = valid.any(1); b = len(valid)
        if self.path != "A":
            patch_memory, patch_valid = self._patch(patches, b, memory.dtype, memory.device)
            if bool((patch_valid.any(1) & ~supported).any()):
                raise ValueError("patch-only support cannot replace empty raw observation")
            memory = torch.cat((memory, patch_memory), 1); memory_valid = torch.cat((memory_valid, patch_valid), 1)
        memory_valid = memory_valid.clone(); memory_valid[~supported, 0] = True
        memory = torch.where(memory_valid[..., None], memory, 0.)
        queries = torch.cat((self.anchor_queries, self.opening_queries), 0)[None].expand(b, -1, -1)
        decoded = self.decoder(queries, memory, memory_key_padding_mask=~memory_valid)
        anchor, opening = decoded[:, :ANCHORS], decoded[:, ANCHORS:]
        direction = self.opening_orientation(opening); norm = torch.linalg.vector_norm(direction, dim=-1)
        direction_valid = (norm > 32 * torch.finfo(direction.dtype).eps) & supported[:, None]
        direction = torch.where(direction_valid[..., None], direction / norm.clamp_min(torch.finfo(direction.dtype).tiny)[..., None], 0.)
        member = self.member_opening(opening) @ self.member_anchor(anchor).transpose(1, 2) / math.sqrt(DIM)
        member_validity = self.validity_opening(opening) @ self.validity_anchor(anchor).transpose(1, 2) / math.sqrt(DIM)
        def observed(value):
            mask = supported.reshape(b, *([1] * (value.ndim - 1)))
            return torch.where(mask, value, 0.)
        return SurfaceRelationPrediction(observed(self.anchor_position(anchor)), observed(self.anchor_presence(anchor).squeeze(-1)),
            observed(F.softplus(self.anchor_uncertainty(anchor))), observed(self.opening_position(opening)),
            observed(self.opening_presence(opening).squeeze(-1)), direction, direction_valid,
            observed(F.softplus(self.opening_dimensions(opening))), observed(self.dimension_evidence(opening)),
            observed(self.opening_support(opening).squeeze(-1)), observed(self.reachability(opening)),
            observed(member), observed(member_validity), supported)

    def forward_compact(self, compact, patches=None):
        from .gse_dual_path_encoder_adapter_v1 import expand_features
        expanded = expand_features(compact)
        return self(expanded.points_xyz_m, expanded.frozen_point_context, expanded.valid, patches,
                    sensor_token_index=expanded.sensor_token_index)


__all__ = ["SurfacePatchBatch", "SurfaceRelationPrediction", "SurfaceRelationModelV1",
           "collate_surface_patches", "REACHABILITY_ORDER"]
