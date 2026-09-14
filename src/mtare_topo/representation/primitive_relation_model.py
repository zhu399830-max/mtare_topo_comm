"""Bottom-up causal LiDAR model for explicit swept primitives and relations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.data.cano_sensor_smoke import lidar_local_directions
from mtare_topo.representation.phase3_structural_semantics import (
    CircularAzimuthConv2d,
    ResidualRangeBlock,
)


HISTORY_FRAMES = 5
ELEVATION_ROWS = 16
AZIMUTH_COLUMNS = 720
MEMORY_AZIMUTH_COLUMNS = 180
MAXIMUM_SLOTS = 32
MODEL_DIM = 128
ENDPOINT_DESCRIPTOR_DIM = 32
MAXIMUM_RANGE_M = 50.0
MINIMUM_SHAPE_EXPONENT = 2.0
MAXIMUM_SHAPE_EXPONENT = 10.0


@dataclass(frozen=True)
class PrimitiveRelationPrediction:
    existence_logits: torch.Tensor
    axis_control_current_sensor_m: torch.Tensor
    endpoint_half_axes_m: torch.Tensor
    endpoint_shape_exponent: torch.Tensor
    endpoint_descriptor: torch.Tensor
    geometry_uncertainty: torch.Tensor
    endpoint_attachment_logits: torch.Tensor
    disconnected_overlap_logits: torch.Tensor
    temporal_correspondence_logits: torch.Tensor
    temporal_presence_logits: torch.Tensor


def _validate_student(
    range_valid: torch.Tensor,
    relative_translation_current_sensor_m: torch.Tensor,
    relative_yaw_current_sensor_deg: torch.Tensor,
) -> None:
    batch = range_valid.shape[0] if range_valid.ndim else -1
    if range_valid.shape != (batch, HISTORY_FRAMES, 2, ELEVATION_ROWS, AZIMUTH_COLUMNS):
        raise ValueError("range_valid must have shape [B,5,2,16,720]")
    if relative_translation_current_sensor_m.shape != (batch, HISTORY_FRAMES, 3):
        raise ValueError("relative translation must have shape [B,5,3]")
    if relative_yaw_current_sensor_deg.shape != (batch, HISTORY_FRAMES):
        raise ValueError("relative yaw must have shape [B,5]")
    values = (range_valid, relative_translation_current_sensor_m, relative_yaw_current_sensor_deg)
    if not all(torch.is_floating_point(value) and bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("student inputs must be finite floating-point tensors")
    ranges, valid = range_valid[:, :, 0], range_valid[:, :, 1]
    if bool(((ranges < 0.0) | (ranges > 1.0)).any()):
        raise ValueError("normalized ranges must be in [0,1]")
    if bool(((valid != 0.0) & (valid != 1.0)).any()):
        raise ValueError("valid-return channel must be binary")
    if not bool((relative_translation_current_sensor_m[:, -1] == 0.0).all()):
        raise ValueError("current-frame relative translation must be exactly zero")
    if not bool((relative_yaw_current_sensor_deg[:, -1] == 0.0).all()):
        raise ValueError("current-frame relative yaw must be exactly zero")


def register_causal_lidar_points(
    range_valid: torch.Tensor,
    relative_translation_current_sensor_m: torch.Tensor,
    relative_yaw_current_sensor_deg: torch.Tensor,
    *, relative_rotation_current_sensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Back-project all five scans into the current sensor coordinate frame."""

    _validate_student(
        range_valid,
        relative_translation_current_sensor_m,
        relative_yaw_current_sensor_deg,
    )
    directions = torch.as_tensor(
        lidar_local_directions(), device=range_valid.device, dtype=range_valid.dtype,
    )
    directions = directions[None, None].expand(len(range_valid), HISTORY_FRAMES, -1, -1, -1)
    yaw = torch.deg2rad(relative_yaw_current_sensor_deg)[:, :, None, None]
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    local_x, local_y, local_z = directions.unbind(dim=-1)
    rotated = torch.stack(
        (cosine * local_x - sine * local_y, sine * local_x + cosine * local_y, local_z),
        dim=-1,
    )
    if relative_rotation_current_sensor is not None:
        rotation = relative_rotation_current_sensor
        if (rotation.shape != (len(range_valid), HISTORY_FRAMES, 3, 3)
                or rotation.dtype != range_valid.dtype or rotation.device != range_valid.device
                or not bool(torch.isfinite(rotation).all())):
            raise ValueError('relative rotations require finite matching [B,5,3,3] tensors')
        eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        if (not torch.allclose(rotation.transpose(-1,-2) @ rotation, eye.expand_as(rotation), atol=1e-6, rtol=0)
                or not torch.allclose(torch.linalg.det(rotation), torch.ones_like(rotation[...,0,0]), atol=1e-6, rtol=0)
                or not torch.equal(rotation[:,-1], eye.expand(len(range_valid),3,3))):
            raise ValueError('proper rotations and exact current identity required')
        # Full rotation governs registered XYZ; the existing yaw embedding
        # remains unchanged, with no new trainable parameters.
        rotated = torch.einsum('bfij,bfrcj->bfrci', rotation, directions)
    ranges_m = range_valid[:, :, 0, ..., None] * MAXIMUM_RANGE_M
    valid = range_valid[:, :, 1].bool()
    points = ranges_m * rotated + relative_translation_current_sensor_m[:, :, None, None, :]
    points = torch.where(valid[..., None], points, torch.zeros_like(points))
    return points, valid


def _token_xyz(points: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(points)
    grouped_points = points.reshape(batch, HISTORY_FRAMES, ELEVATION_ROWS, MEMORY_AZIMUTH_COLUMNS, 4, 3)
    grouped_valid = valid.reshape(batch, HISTORY_FRAMES, ELEVATION_ROWS, MEMORY_AZIMUTH_COLUMNS, 4)
    weights = grouped_valid[..., None].to(points.dtype)
    count = weights.sum(dim=(2, 4)).clamp_min(1.0)
    xyz = (grouped_points * weights).sum(dim=(2, 4)) / count
    token_valid = grouped_valid.any(dim=(2, 4))
    return xyz, token_valid


class PrimitiveRelationNet(nn.Module):
    """Predict a permutation-safe 32-slot primitive set and its relations.

    Only range/valid images and causal relative odometry enter ``forward``.
    Registered point coordinates are produced by the deterministic sensor
    contract above; world, graph, traversal and construction identities are
    deliberately absent from the interface.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            CircularAzimuthConv2d(5, 32, stride=(1, 2)),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            ResidualRangeBlock(32, 64, (2, 2)),
            ResidualRangeBlock(64, 96, (2, 1)),
            ResidualRangeBlock(96, MODEL_DIM, (2, 1)),
            ResidualRangeBlock(MODEL_DIM, MODEL_DIM, (2, 1)),
        )
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, MODEL_DIM),
        )
        self.pose_invariant_embedding = nn.Sequential(
            nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, MODEL_DIM),
        )
        self.time_embedding = nn.Embedding(HISTORY_FRAMES, MODEL_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            MODEL_DIM, 8, 4 * MODEL_DIM, dropout=0.0, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2, norm=nn.LayerNorm(MODEL_DIM),
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            MODEL_DIM, 8, 4 * MODEL_DIM, dropout=0.0, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.slot_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=2, norm=nn.LayerNorm(MODEL_DIM),
        )
        self.slot_query = nn.Parameter(torch.empty(MAXIMUM_SLOTS, MODEL_DIM))
        nn.init.normal_(self.slot_query, std=0.02)
        self.control_query = nn.Linear(MODEL_DIM, 3 * MODEL_DIM)
        self.existence_head = nn.Linear(MODEL_DIM, 1)
        self.shape_head = nn.Linear(MODEL_DIM, 6)
        self.descriptor_head = nn.Linear(MODEL_DIM, 2 * ENDPOINT_DESCRIPTOR_DIM)
        self.uncertainty_head = nn.Linear(MODEL_DIM, 1)
        endpoint_feature_dim = MODEL_DIM + ENDPOINT_DESCRIPTOR_DIM + 3
        self.attachment_head = nn.Sequential(
            nn.Linear(2 * endpoint_feature_dim + 1, MODEL_DIM),
            nn.SiLU(), nn.Linear(MODEL_DIM, 1),
        )
        primitive_feature_dim = MODEL_DIM + 3
        self.overlap_head = nn.Sequential(
            nn.Linear(2 * primitive_feature_dim + 1, MODEL_DIM),
            nn.SiLU(), nn.Linear(MODEL_DIM, 1),
        )
        self.frame_attention = nn.MultiheadAttention(
            MODEL_DIM, 8, dropout=0.0, batch_first=True,
        )
        self.temporal_source = nn.Linear(MODEL_DIM, MODEL_DIM, bias=False)
        self.temporal_destination = nn.Linear(MODEL_DIM, MODEL_DIM, bias=False)
        self.temporal_dustbin = nn.Linear(MODEL_DIM, 1)
        self.temporal_presence = nn.Linear(MODEL_DIM, 1)

    def _memory(
        self,
        range_valid: torch.Tensor,
        translation: torch.Tensor,
        yaw_deg: torch.Tensor,
        relative_rotation_current_sensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        points, valid = register_causal_lidar_points(range_valid, translation, yaw_deg,
            relative_rotation_current_sensor=relative_rotation_current_sensor)
        token_xyz, token_valid = _token_xyz(points, valid)
        if bool((~token_valid).all(dim=-1).any()):
            raise ValueError("every causal frame must contain at least one valid LiDAR token")
        batch = len(range_valid)
        registered = points / MAXIMUM_RANGE_M
        encoder_input = torch.cat(
            (range_valid, registered.permute(0, 1, 4, 2, 3)), dim=2,
        ).reshape(batch * HISTORY_FRAMES, 5, ELEVATION_ROWS, AZIMUTH_COLUMNS)
        encoded = self.encoder(encoder_input)
        if encoded.shape[2:] != (1, MEMORY_AZIMUTH_COLUMNS):
            raise RuntimeError(f"primitive encoder resolution drift: {tuple(encoded.shape)}")
        memory = encoded.squeeze(2).transpose(1, 2).reshape(
            batch, HISTORY_FRAMES, MEMORY_AZIMUTH_COLUMNS, MODEL_DIM,
        )
        coordinate = self.coordinate_embedding(token_xyz / MAXIMUM_RANGE_M)
        yaw = torch.deg2rad(yaw_deg)
        pose_invariant = torch.stack(
            (
                torch.linalg.vector_norm(translation[..., :2], dim=-1) / MAXIMUM_RANGE_M,
                translation[..., 2] / MAXIMUM_RANGE_M,
                torch.cos(yaw), torch.sin(yaw),
                torch.linalg.vector_norm(translation, dim=-1) / MAXIMUM_RANGE_M,
            ), dim=-1,
        )
        pose = self.pose_invariant_embedding(pose_invariant)[:, :, None]
        time = self.time_embedding(torch.arange(HISTORY_FRAMES, device=range_valid.device))[None, :, None]
        frame_memory = memory + coordinate + pose + time
        flat_memory = frame_memory.reshape(batch, -1, MODEL_DIM)
        flat_valid = token_valid.reshape(batch, -1)
        fused = self.temporal_encoder(flat_memory, src_key_padding_mask=~flat_valid)
        return fused, flat_valid, token_xyz.reshape(batch, -1, 3), frame_memory

    @staticmethod
    def _symmetric_pair_features(features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        first, second = features[:, :, None], features[:, None, :]
        distance = torch.linalg.vector_norm(positions[:, :, None] - positions[:, None, :], dim=-1, keepdim=True) / MAXIMUM_RANGE_M
        return torch.cat((first + second, torch.abs(first - second), distance), dim=-1)

    def forward(
        self,
        range_valid: torch.Tensor,
        relative_translation_current_sensor_m: torch.Tensor,
        relative_yaw_current_sensor_deg: torch.Tensor,
        *,
        query_permutation: torch.Tensor | None = None,
        relative_rotation_current_sensor: torch.Tensor | None = None,
    ) -> PrimitiveRelationPrediction:
        _validate_student(
            range_valid,
            relative_translation_current_sensor_m,
            relative_yaw_current_sensor_deg,
        )
        memory, memory_valid, memory_xyz, frame_memory = self._memory(
            range_valid,
            relative_translation_current_sensor_m,
            relative_yaw_current_sensor_deg,
            relative_rotation_current_sensor,
        )
        query = self.slot_query
        if query_permutation is not None:
            if query_permutation.shape != (MAXIMUM_SLOTS,) or query_permutation.dtype != torch.long:
                raise ValueError("query permutation must be int64 [32]")
            if not torch.equal(torch.sort(query_permutation).values, torch.arange(MAXIMUM_SLOTS, device=query_permutation.device)):
                raise ValueError("query permutation must be a bijection")
            query = query[query_permutation]
        query = query[None].expand(len(range_valid), -1, -1)
        slots = self.slot_decoder(query, memory, memory_key_padding_mask=~memory_valid)

        control_queries = self.control_query(slots).reshape(
            len(range_valid), MAXIMUM_SLOTS, 3, MODEL_DIM,
        )
        control_logits = torch.einsum("bscd,bnd->bscn", control_queries, memory) / math.sqrt(MODEL_DIM)
        control_logits = control_logits.masked_fill(~memory_valid[:, None, None], -torch.inf)
        control_weight = torch.softmax(control_logits, dim=-1)
        axis_control = torch.einsum("bscn,bnd->bscd", control_weight, memory_xyz)

        shape = self.shape_head(slots)
        half_axes = F.softplus(shape[..., :4]).reshape(len(range_valid), MAXIMUM_SLOTS, 2, 2) + 1e-4
        exponent = MINIMUM_SHAPE_EXPONENT + (MAXIMUM_SHAPE_EXPONENT - MINIMUM_SHAPE_EXPONENT) * torch.sigmoid(shape[..., 4:6])
        descriptor = F.normalize(
            self.descriptor_head(slots).reshape(len(range_valid), MAXIMUM_SLOTS, 2, ENDPOINT_DESCRIPTOR_DIM),
            dim=-1, eps=1e-8,
        )
        uncertainty = F.softplus(self.uncertainty_head(slots).squeeze(-1)) + 1e-4

        endpoints = axis_control[:, :, (0, 2)]
        endpoint_features = torch.cat(
            (
                slots[:, :, None].expand(-1, -1, 2, -1),
                descriptor,
                endpoints / MAXIMUM_RANGE_M,
            ), dim=-1,
        ).reshape(len(range_valid), 2 * MAXIMUM_SLOTS, -1)
        endpoint_positions = endpoints.reshape(len(range_valid), 2 * MAXIMUM_SLOTS, 3)
        attachment = self.attachment_head(
            self._symmetric_pair_features(endpoint_features, endpoint_positions),
        ).squeeze(-1)
        attachment = 0.5 * (attachment + attachment.transpose(1, 2))
        attachment = attachment.reshape(len(range_valid), MAXIMUM_SLOTS, 2, MAXIMUM_SLOTS, 2)

        centers = axis_control.mean(dim=2)
        primitive_features = torch.cat((slots, centers / MAXIMUM_RANGE_M), dim=-1)
        overlap = self.overlap_head(
            self._symmetric_pair_features(primitive_features, centers),
        ).squeeze(-1)
        overlap = 0.5 * (overlap + overlap.transpose(1, 2))

        batch = len(range_valid)
        frame_flat = frame_memory.reshape(batch * HISTORY_FRAMES, MEMORY_AZIMUTH_COLUMNS, MODEL_DIM)
        frame_mask = ~(range_valid[:, :, 1].reshape(batch, HISTORY_FRAMES, ELEVATION_ROWS, MEMORY_AZIMUTH_COLUMNS, 4).bool().any(dim=(2, 4)))
        frame_query = slots[:, None].expand(-1, HISTORY_FRAMES, -1, -1).reshape(batch * HISTORY_FRAMES, MAXIMUM_SLOTS, MODEL_DIM)
        frame_slots, _ = self.frame_attention(
            frame_query, frame_flat, frame_flat,
            key_padding_mask=frame_mask.reshape(batch * HISTORY_FRAMES, MEMORY_AZIMUTH_COLUMNS),
            need_weights=False,
        )
        frame_slots = frame_slots.reshape(batch, HISTORY_FRAMES, MAXIMUM_SLOTS, MODEL_DIM)
        source = self.temporal_source(frame_slots)
        destination = self.temporal_destination(slots)
        correspondence = torch.einsum("btsd,bqd->btsq", source, destination) / math.sqrt(MODEL_DIM)
        dustbin = self.temporal_dustbin(frame_slots)
        temporal = torch.cat((correspondence, dustbin), dim=-1)

        return PrimitiveRelationPrediction(
            existence_logits=self.existence_head(slots).squeeze(-1),
            axis_control_current_sensor_m=axis_control,
            endpoint_half_axes_m=half_axes,
            endpoint_shape_exponent=exponent,
            endpoint_descriptor=descriptor,
            geometry_uncertainty=uncertainty,
            endpoint_attachment_logits=attachment,
            disconnected_overlap_logits=overlap,
            temporal_correspondence_logits=temporal,
            temporal_presence_logits=self.temporal_presence(frame_slots).squeeze(-1),
        )


__all__ = [
    "AZIMUTH_COLUMNS", "ENDPOINT_DESCRIPTOR_DIM", "ELEVATION_ROWS",
    "HISTORY_FRAMES", "MAXIMUM_RANGE_M", "MAXIMUM_SLOTS", "MODEL_DIM",
    "PrimitiveRelationNet", "PrimitiveRelationPrediction",
    "register_causal_lidar_points",
]
