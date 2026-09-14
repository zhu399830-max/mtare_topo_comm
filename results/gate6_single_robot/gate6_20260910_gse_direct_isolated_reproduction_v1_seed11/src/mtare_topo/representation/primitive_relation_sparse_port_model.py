"""Sparse set and endpoint-context relation corrective for primitive V1.

The V1 C07 attribution showed two independent faults: almost every one of the
32 set slots was active, and the frozen pair MLP could not rank a true
attachment at the required safe precision even under a Teacher proposal
oracle.  This module preserves the causal LiDAR encoder and swept-geometry
interface while replacing only the evidenced set/relation bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.representation.primitive_relation_model import (
    ENDPOINT_DESCRIPTOR_DIM,
    HISTORY_FRAMES,
    MAXIMUM_RANGE_M,
    MAXIMUM_SHAPE_EXPONENT,
    MAXIMUM_SLOTS,
    MINIMUM_SHAPE_EXPONENT,
    MODEL_DIM,
    PrimitiveRelationNet,
    _validate_student,
)


PAIR_GEOMETRY_DIM = 9


@dataclass(frozen=True)
class SparsePortRelationPrediction:
    existence_logits: torch.Tensor
    axis_control_current_sensor_m: torch.Tensor
    endpoint_half_axes_m: torch.Tensor
    endpoint_shape_exponent: torch.Tensor
    endpoint_descriptor: torch.Tensor
    geometry_uncertainty: torch.Tensor
    endpoint_attachment_logits: torch.Tensor
    endpoint_attachment_uncertainty: torch.Tensor
    disconnected_overlap_logits: torch.Tensor
    disconnected_overlap_uncertainty: torch.Tensor
    temporal_correspondence_logits: torch.Tensor
    temporal_presence_logits: torch.Tensor


def outward_endpoint_tangents(axis_control: torch.Tensor) -> torch.Tensor:
    """Return endpoint tangents directed from the primitive interior outward."""

    if axis_control.shape[-2:] != (3, 3):
        raise ValueError("axis controls must end in [3,3]")
    start = F.normalize(axis_control[..., 0, :] - axis_control[..., 1, :], dim=-1, eps=1e-6)
    finish = F.normalize(axis_control[..., 2, :] - axis_control[..., 1, :], dim=-1, eps=1e-6)
    return torch.stack((start, finish), dim=-2)


def invariant_endpoint_pair_geometry(
    positions_m: torch.Tensor,
    outward_tangent: torch.Tensor,
    half_axes_m: torch.Tensor,
    exponent: torch.Tensor,
    descriptor: torch.Tensor,
) -> torch.Tensor:
    """Construct symmetric rotation-invariant pair geometry.

    No hand-written attachment threshold is applied.  These quantities are
    inputs to the learned relation decoder and remain meaningful for rotated,
    sloped and stacked tunnels.
    """

    count = positions_m.shape[-2]
    expected = positions_m.shape[:-1]
    if outward_tangent.shape != positions_m.shape:
        raise ValueError("pair tangents must match positions")
    if half_axes_m.shape != (*expected, 2):
        raise ValueError("pair half axes must end in [N,2]")
    if exponent.shape != expected:
        raise ValueError("pair exponent must end in [N]")
    if descriptor.shape[:-1] != expected:
        raise ValueError("pair descriptors must share the item axis")
    if count < 1:
        raise ValueError("at least one endpoint is required")

    delta = positions_m[..., None, :, :] - positions_m[..., :, None, :]
    distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    direction = delta / distance.clamp_min(1e-6)
    first_tangent = outward_tangent[..., :, None, :]
    second_tangent = outward_tangent[..., None, :, :]
    tangent_dot = (first_tangent * second_tangent).sum(dim=-1, keepdim=True)
    # Symmetric under exchanging endpoints: both outward tangents should face
    # the other endpoint for a physical attachment.
    facing = 0.5 * (
        (first_tangent * direction).sum(dim=-1, keepdim=True)
        - (second_tangent * direction).sum(dim=-1, keepdim=True)
    )
    vertical = delta[..., 2:3].abs() / MAXIMUM_RANGE_M
    horizontal = torch.linalg.vector_norm(delta[..., :2], dim=-1, keepdim=True) / MAXIMUM_RANGE_M
    section = torch.abs(
        torch.log1p(half_axes_m[..., :, None, :])
        - torch.log1p(half_axes_m[..., None, :, :])
    ) / math.log(11.0)
    exponent_difference = torch.abs(
        exponent[..., :, None] - exponent[..., None, :]
    )[..., None] / (MAXIMUM_SHAPE_EXPONENT - MINIMUM_SHAPE_EXPONENT)
    descriptor_cosine = (
        F.normalize(descriptor, dim=-1, eps=1e-8)[..., :, None, :]
        * F.normalize(descriptor, dim=-1, eps=1e-8)[..., None, :, :]
    ).sum(dim=-1, keepdim=True)
    result = torch.cat(
        (
            distance / MAXIMUM_RANGE_M,
            tangent_dot,
            facing,
            vertical,
            horizontal,
            section,
            exponent_difference,
            descriptor_cosine,
        ),
        dim=-1,
    )
    if result.shape[-1] != PAIR_GEOMETRY_DIM:
        raise RuntimeError("pair geometry feature dimension drift")
    return result


def _symmetric_context_pair_features(
    context: torch.Tensor,
    geometry: torch.Tensor,
) -> torch.Tensor:
    first, second = context[:, :, None], context[:, None, :]
    return torch.cat((first + second, torch.abs(first - second), geometry), dim=-1)


class SparsePortRelationNet(PrimitiveRelationNet):
    """V2 set decoder with endpoint relation context and typed uncertainty."""

    def __init__(self) -> None:
        super().__init__()
        # Remove the V1 dense pair MLPs so every V2 parameter participates in
        # the new objective and frozen V1 checkpoints remain a separate type.
        del self.attachment_head
        del self.overlap_head

        endpoint_input = (
            MODEL_DIM + ENDPOINT_DESCRIPTOR_DIM + 3 + 3 + 2 + 1 + 1 + 1
        )
        self.endpoint_token_embedding = nn.Sequential(
            nn.Linear(endpoint_input, MODEL_DIM), nn.SiLU(), nn.Linear(MODEL_DIM, MODEL_DIM),
        )
        endpoint_layer = nn.TransformerEncoderLayer(
            MODEL_DIM, 8, 4 * MODEL_DIM, dropout=0.0, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.endpoint_relation_transformer = nn.TransformerEncoder(
            endpoint_layer, num_layers=2, norm=nn.LayerNorm(MODEL_DIM),
            enable_nested_tensor=False,
        )
        pair_input = 2 * MODEL_DIM + PAIR_GEOMETRY_DIM
        self.attachment_relation_head = nn.Sequential(
            nn.Linear(pair_input, MODEL_DIM), nn.SiLU(), nn.Linear(MODEL_DIM, 1),
        )
        self.attachment_uncertainty_head = nn.Sequential(
            nn.Linear(pair_input, MODEL_DIM // 2), nn.SiLU(), nn.Linear(MODEL_DIM // 2, 1),
        )
        primitive_layer = nn.TransformerEncoderLayer(
            MODEL_DIM, 8, 4 * MODEL_DIM, dropout=0.0, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.primitive_relation_transformer = nn.TransformerEncoder(
            primitive_layer, num_layers=1, norm=nn.LayerNorm(MODEL_DIM),
            enable_nested_tensor=False,
        )
        self.overlap_relation_head = nn.Sequential(
            nn.Linear(pair_input, MODEL_DIM), nn.SiLU(), nn.Linear(MODEL_DIM, 1),
        )
        self.overlap_uncertainty_head = nn.Sequential(
            nn.Linear(pair_input, MODEL_DIM // 2), nn.SiLU(), nn.Linear(MODEL_DIM // 2, 1),
        )

    @staticmethod
    def _symmetric(value: torch.Tensor) -> torch.Tensor:
        return 0.5 * (value + value.transpose(1, 2))

    def forward(
        self,
        range_valid: torch.Tensor,
        relative_translation_current_sensor_m: torch.Tensor,
        relative_yaw_current_sensor_deg: torch.Tensor,
        *,
        query_permutation: torch.Tensor | None = None,
    ) -> SparsePortRelationPrediction:
        _validate_student(
            range_valid,
            relative_translation_current_sensor_m,
            relative_yaw_current_sensor_deg,
        )
        memory, memory_valid, memory_xyz, frame_memory = self._memory(
            range_valid,
            relative_translation_current_sensor_m,
            relative_yaw_current_sensor_deg,
        )
        query = self.slot_query
        if query_permutation is not None:
            if query_permutation.shape != (MAXIMUM_SLOTS,) or query_permutation.dtype != torch.long:
                raise ValueError("query permutation must be int64 [32]")
            if not torch.equal(
                torch.sort(query_permutation).values,
                torch.arange(MAXIMUM_SLOTS, device=query_permutation.device),
            ):
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
        half_axes = F.softplus(shape[..., :4]).reshape(
            len(range_valid), MAXIMUM_SLOTS, 2, 2,
        ) + 1e-4
        exponent = MINIMUM_SHAPE_EXPONENT + (
            MAXIMUM_SHAPE_EXPONENT - MINIMUM_SHAPE_EXPONENT
        ) * torch.sigmoid(shape[..., 4:6])
        descriptor = F.normalize(
            self.descriptor_head(slots).reshape(
                len(range_valid), MAXIMUM_SLOTS, 2, ENDPOINT_DESCRIPTOR_DIM,
            ),
            dim=-1,
            eps=1e-8,
        )
        geometry_uncertainty = F.softplus(self.uncertainty_head(slots).squeeze(-1)) + 1e-4
        existence_logits = self.existence_head(slots).squeeze(-1)
        existence_probability = torch.sigmoid(existence_logits)

        endpoints = axis_control[:, :, (0, 2)]
        tangent = outward_endpoint_tangents(axis_control)
        endpoint_raw = torch.cat(
            (
                slots[:, :, None].expand(-1, -1, 2, -1),
                descriptor,
                endpoints / MAXIMUM_RANGE_M,
                tangent,
                torch.log1p(half_axes) / math.log(11.0),
                exponent[..., None],
                geometry_uncertainty[:, :, None, None].expand(-1, -1, 2, -1),
                existence_probability[:, :, None, None].expand(-1, -1, 2, -1),
            ),
            dim=-1,
        )
        endpoint_token = self.endpoint_token_embedding(endpoint_raw)
        endpoint_token = endpoint_token * (
            0.05 + 0.95 * existence_probability[:, :, None, None]
        )
        endpoint_context = self.endpoint_relation_transformer(
            endpoint_token.reshape(len(range_valid), 2 * MAXIMUM_SLOTS, MODEL_DIM),
        )
        endpoint_positions = endpoints.reshape(len(range_valid), 2 * MAXIMUM_SLOTS, 3)
        endpoint_tangent = tangent.reshape(len(range_valid), 2 * MAXIMUM_SLOTS, 3)
        endpoint_axes = half_axes.reshape(len(range_valid), 2 * MAXIMUM_SLOTS, 2)
        endpoint_exponent = exponent.reshape(len(range_valid), 2 * MAXIMUM_SLOTS)
        endpoint_descriptor = descriptor.reshape(
            len(range_valid), 2 * MAXIMUM_SLOTS, ENDPOINT_DESCRIPTOR_DIM,
        )
        attachment_geometry = invariant_endpoint_pair_geometry(
            endpoint_positions,
            endpoint_tangent,
            endpoint_axes,
            endpoint_exponent,
            endpoint_descriptor,
        )
        attachment_features = _symmetric_context_pair_features(
            endpoint_context, attachment_geometry,
        )
        attachment = self._symmetric(
            self.attachment_relation_head(attachment_features).squeeze(-1),
        )
        attachment_uncertainty = self._symmetric(
            torch.sigmoid(self.attachment_uncertainty_head(attachment_features).squeeze(-1)),
        )
        attachment = attachment.reshape(
            len(range_valid), MAXIMUM_SLOTS, 2, MAXIMUM_SLOTS, 2,
        )
        attachment_uncertainty = attachment_uncertainty.reshape(
            len(range_valid), MAXIMUM_SLOTS, 2, MAXIMUM_SLOTS, 2,
        )

        primitive_context = self.primitive_relation_transformer(
            endpoint_context.reshape(len(range_valid), MAXIMUM_SLOTS, 2, MODEL_DIM).mean(dim=2),
        )
        centers = axis_control.mean(dim=2)
        primitive_tangent = F.normalize(
            axis_control[:, :, 2] - axis_control[:, :, 0], dim=-1, eps=1e-6,
        )
        primitive_geometry = invariant_endpoint_pair_geometry(
            centers,
            primitive_tangent,
            half_axes.mean(dim=2),
            exponent.mean(dim=2),
            F.normalize(descriptor.mean(dim=2), dim=-1, eps=1e-8),
        )
        overlap_features = _symmetric_context_pair_features(
            primitive_context, primitive_geometry,
        )
        overlap = self._symmetric(
            self.overlap_relation_head(overlap_features).squeeze(-1),
        )
        overlap_uncertainty = self._symmetric(
            torch.sigmoid(self.overlap_uncertainty_head(overlap_features).squeeze(-1)),
        )

        batch = len(range_valid)
        frame_flat = frame_memory.reshape(
            batch * HISTORY_FRAMES, -1, MODEL_DIM,
        )
        frame_mask = ~(
            range_valid[:, :, 1]
            .reshape(batch, HISTORY_FRAMES, 16, -1, 4)
            .bool()
            .any(dim=(2, 4))
        )
        frame_query = slots[:, None].expand(
            -1, HISTORY_FRAMES, -1, -1,
        ).reshape(batch * HISTORY_FRAMES, MAXIMUM_SLOTS, MODEL_DIM)
        frame_slots, _ = self.frame_attention(
            frame_query,
            frame_flat,
            frame_flat,
            key_padding_mask=frame_mask.reshape(batch * HISTORY_FRAMES, -1),
            need_weights=False,
        )
        frame_slots = frame_slots.reshape(
            batch, HISTORY_FRAMES, MAXIMUM_SLOTS, MODEL_DIM,
        )
        source = self.temporal_source(frame_slots)
        destination = self.temporal_destination(slots)
        correspondence = torch.einsum(
            "btsd,bqd->btsq", source, destination,
        ) / math.sqrt(MODEL_DIM)
        temporal = torch.cat((correspondence, self.temporal_dustbin(frame_slots)), dim=-1)

        return SparsePortRelationPrediction(
            existence_logits=existence_logits,
            axis_control_current_sensor_m=axis_control,
            endpoint_half_axes_m=half_axes,
            endpoint_shape_exponent=exponent,
            endpoint_descriptor=descriptor,
            geometry_uncertainty=geometry_uncertainty,
            endpoint_attachment_logits=attachment,
            endpoint_attachment_uncertainty=attachment_uncertainty,
            disconnected_overlap_logits=overlap,
            disconnected_overlap_uncertainty=overlap_uncertainty,
            temporal_correspondence_logits=temporal,
            temporal_presence_logits=self.temporal_presence(frame_slots).squeeze(-1),
        )


__all__ = [
    "PAIR_GEOMETRY_DIM",
    "SparsePortRelationNet",
    "SparsePortRelationPrediction",
    "invariant_endpoint_pair_geometry",
    "outward_endpoint_tangents",
]
