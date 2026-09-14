"""Axis-anchored geometry-semantic event relations for GSE-Graph.

The network consumes only five causal organized LiDAR scans.  It predicts a
five-way structural event together with per-bearing persistent/reveal/withdraw
relations.  Teacher identities are accepted by the loss only; they are never
forward inputs.
"""

from __future__ import annotations

import inspect
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.representation.phase3_structural_semantics import (
    CircularAzimuthConv2d,
    ResidualRangeBlock,
)


HISTORY = 5
ELEVATION_ROWS = 16
AZIMUTH_COLUMNS = 720
BEARING_BINS = 180
EVENT_CLASSES = 5
RELATION_CLASSES = 3
RELATION_NAMES = ("persistent", "reveal", "withdraw")
ENCODER_DIM = 128
PLACE_DESCRIPTOR_DIM = 128
BRANCH_DESCRIPTOR_DIM = 32
PROFILE_DIM = 4
HEADING_RESIDUAL_LIMIT_DEG = 1.0
PROFILE_SCALE_M = 5.0
METRIC_DISTANCE_SCALE_M = 30.0
SLOPE_SCALE_DEG = 45.0
CURVATURE_SCALE_PER_M = 0.1


class AxisAnchoredEventRelationNet(nn.Module):
    """Learn structural events and branch relations in the sensor-forward frame."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            CircularAzimuthConv2d(2, 32, stride=(1, 2)),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            ResidualRangeBlock(32, 64, (2, 2)),
            ResidualRangeBlock(64, 96, (2, 1)),
            ResidualRangeBlock(96, ENCODER_DIM, (2, 1)),
        )
        self.global_temporal = nn.GRU(ENCODER_DIM, ENCODER_DIM, batch_first=True)
        self.event_head = nn.Linear(ENCODER_DIM, EVENT_CLASSES)
        self.axis_azimuth_head = nn.Sequential(
            nn.Conv1d(ENCODER_DIM, 64, 1), nn.SiLU(), nn.Conv1d(64, 1, 1)
        )
        self.axis_vertical_head = nn.Linear(ENCODER_DIM, 1)
        self.geometry_head = nn.Linear(ENCODER_DIM, 4)
        self.place_head = nn.Sequential(
            nn.Linear(ENCODER_DIM, ENCODER_DIM), nn.SiLU(),
            nn.Linear(ENCODER_DIM, PLACE_DESCRIPTOR_DIM),
        )
        self.observation_uncertainty_head = nn.Linear(ENCODER_DIM, 1)

        relation_input = 4 * ENCODER_DIM
        relation_values = RELATION_CLASSES + 1 + 1 + PROFILE_DIM + BRANCH_DESCRIPTOR_DIM + 6
        self.relation_head = nn.Sequential(
            nn.Conv1d(relation_input, ENCODER_DIM, 3, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(ENCODER_DIM, relation_values, 1),
        )
        azimuth = torch.arange(BEARING_BINS, dtype=torch.float32) * (
            2.0 * torch.pi / BEARING_BINS
        )
        self.register_buffer("bearing_azimuth_rad", azimuth, persistent=True)

    @staticmethod
    def _validate_scans(scans: torch.Tensor) -> None:
        expected = (HISTORY, 2, ELEVATION_ROWS, AZIMUTH_COLUMNS)
        if scans.ndim != 5 or tuple(scans.shape[1:]) != expected:
            raise ValueError(f"expected [B,5,2,16,720], got {tuple(scans.shape)}")
        if not torch.is_floating_point(scans) or not bool(torch.isfinite(scans).all()):
            raise ValueError("axis-anchored input must be finite floating point")
        ranges = scans[:, :, 0]
        valid = scans[:, :, 1]
        if bool((ranges < 0.0).any()) or bool((ranges > 1.0).any()):
            raise ValueError("normalized range channel must be in [0,1]")
        if bool(((valid != 0.0) & (valid != 1.0)).any()):
            raise ValueError("valid-return channel must be binary")

    def _encode(self, scans: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_scans(scans)
        batch = len(scans)
        encoded = self.encoder(scans.reshape(batch * HISTORY, 2, ELEVATION_ROWS, AZIMUTH_COLUMNS))
        if encoded.shape[-1] != BEARING_BINS:
            raise RuntimeError("axis-anchored bearing resolution drift")
        directional = encoded.mean(dim=2).reshape(batch, HISTORY, ENCODER_DIM, BEARING_BINS)
        pooled = directional.mean(dim=-1)
        return directional, pooled

    def forward(self, scans: torch.Tensor) -> dict[str, torch.Tensor]:
        directional, pooled = self._encode(scans)
        temporal, _ = self.global_temporal(pooled)
        context = temporal[:, -1]

        pair_features = []
        for index in range(1, HISTORY):
            previous = directional[:, index - 1]
            current = directional[:, index]
            past_mean = directional[:, :index].mean(dim=1)
            pair_features.append(torch.cat((current, previous, current - previous, past_mean), dim=1))
        raw = torch.stack([self.relation_head(value) for value in pair_features], dim=1).transpose(2, 3)
        offset = 0
        relation_logits = raw[..., offset : offset + RELATION_CLASSES]
        offset += RELATION_CLASSES
        heading_residual = HEADING_RESIDUAL_LIMIT_DEG * torch.tanh(raw[..., offset])
        offset += 1
        opening_width = F.softplus(raw[..., offset]) + 1e-4
        offset += 1
        vertical_profile = PROFILE_SCALE_M * torch.tanh(raw[..., offset : offset + PROFILE_DIM])
        offset += PROFILE_DIM
        branch_descriptor = F.normalize(
            raw[..., offset : offset + BRANCH_DESCRIPTOR_DIM], dim=-1, eps=1e-8
        )
        offset += BRANCH_DESCRIPTOR_DIM
        branch_uncertainty = F.softplus(raw[..., offset : offset + 6]) + 0.05

        current_directional = directional[:, -1]
        azimuth = self.bearing_azimuth_rad.to(dtype=scans.dtype)
        axis_probability = torch.softmax(
            self.axis_azimuth_head(current_directional).squeeze(1), dim=-1
        )
        horizontal_axis = F.normalize(
            torch.stack((
                (axis_probability * torch.cos(azimuth)).sum(dim=-1),
                (axis_probability * torch.sin(azimuth)).sum(dim=-1),
            ), dim=-1), dim=-1, eps=1e-8,
        )
        local_axis = F.normalize(
            torch.cat((horizontal_axis, torch.tanh(self.axis_vertical_head(context))), dim=-1),
            dim=-1, eps=1e-8,
        )
        geometry = self.geometry_head(context)
        relation_probability = torch.sigmoid(relation_logits)
        branch_union_probability = causal_branch_union_probability(relation_probability)
        event_logits = self.event_head(context)
        return {
            "event_logits": event_logits,
            "event_probability": torch.softmax(event_logits, dim=-1),
            "relation_logits_sequence": relation_logits,
            "relation_probability_sequence": relation_probability,
            "branch_relation_logits": relation_logits[:, -1],
            "branch_relation_probability": relation_probability[:, -1],
            "branch_union_probability": branch_union_probability,
            "branch_heading_residual_deg": heading_residual[:, -1],
            "branch_opening_width_m": opening_width[:, -1],
            "branch_vertical_profile_m": vertical_profile[:, -1],
            "branch_descriptor": branch_descriptor[:, -1],
            "branch_geometry_uncertainty": branch_uncertainty[:, -1],
            "local_axis": local_axis,
            "width_m": F.softplus(geometry[:, 0]) + 1e-4,
            "height_m": F.softplus(geometry[:, 1]) + 1e-4,
            "slope_deg": SLOPE_SCALE_DEG * torch.tanh(geometry[:, 2]),
            "curvature_per_m": F.softplus(geometry[:, 3]),
            "place_descriptor": F.normalize(self.place_head(context), dim=-1, eps=1e-8),
            "observation_uncertainty": torch.sigmoid(
                self.observation_uncertainty_head(context).squeeze(-1)
            ),
        }


def causal_branch_union_probability(relation_probability: torch.Tensor) -> torch.Tensor:
    """Fold adjacent-frame relations using past-to-current state only."""
    if relation_probability.ndim != 4 or tuple(relation_probability.shape[1:]) != (
        HISTORY - 1, BEARING_BINS, RELATION_CLASSES,
    ):
        raise ValueError("relation probability must be [B,4,180,3]")
    if (
        not bool(torch.isfinite(relation_probability).all())
        or bool((relation_probability < 0).any())
        or bool((relation_probability > 1).any())
    ):
        raise ValueError("relation probability must be finite and in [0,1]")
    occupancy = (
        relation_probability[:, 0, :, 1]
        + (1.0 - relation_probability[:, 0, :, 1]) * relation_probability[:, 0, :, 0]
    )
    for index in range(1, HISTORY - 1):
        persistent = relation_probability[:, index, :, 0]
        reveal = relation_probability[:, index, :, 1]
        withdraw = relation_probability[:, index, :, 2]
        # A newly revealed branch can replace a withdrawn old branch in the
        # same 2-degree bin; reveal therefore owns current occupancy.
        occupancy = reveal + (1.0 - reveal) * persistent * occupancy * (1.0 - withdraw)
    return occupancy.clamp(0.0, 1.0)


def reverse_axis_field(values: torch.Tensor, *, bearing_dimension: int = -1) -> torch.Tensor:
    """Express a circular field in the opposite traversal-axis frame."""
    if values.shape[bearing_dimension] != BEARING_BINS:
        raise ValueError("reverse-axis field requires 180 bearing bins")
    return torch.roll(values, shifts=BEARING_BINS // 2, dims=bearing_dimension)


def _balanced_relation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_rate: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape != target.shape or logits.shape[-1] != RELATION_CLASSES:
        raise ValueError("relation target shape drift")
    if torch.any((target < -1) | (target > 1)):
        raise ValueError("multi-label relation target outside -1/0/1")
    valid = target >= 0
    if not bool(valid.any()):
        raise ValueError("relation target contains no valid past-to-current pair")
    if positive_rate is not None:
        rate = positive_rate.to(device=logits.device, dtype=logits.dtype)
        if rate.shape != (RELATION_CLASSES,) or not bool(torch.isfinite(rate).all()) or bool((rate <= 0).any()) or bool((rate >= 1).any()):
            raise ValueError("relation positive rate must contain three values in (0,1)")
        element = F.binary_cross_entropy_with_logits(logits, target.clamp_min(0).to(logits.dtype), reduction="none")
        weight = torch.where(target == 1, 0.5 / rate, 0.5 / (1.0 - rate))
        weight = torch.where(valid, weight, torch.zeros_like(weight))
        return (element * weight).sum() / weight.sum().clamp_min(1e-8)
    terms = []
    for relation in range(RELATION_CLASSES):
        relation_valid = valid[..., relation]
        positive = relation_valid & (target[..., relation] == 1)
        negative = relation_valid & (target[..., relation] == 0)
        if bool(positive.any()) and bool(negative.any()):
            terms.append(0.5 * (
                F.softplus(-logits[..., relation][positive]).mean()
                + F.softplus(logits[..., relation][negative]).mean()
            ))
    if len(terms) != RELATION_CLASSES:
        raise ValueError("real relation batch must cover positive and negative persistent/reveal/withdraw")
    return torch.stack(terms).mean()


def _identity_contrastive(descriptor: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    if descriptor.ndim != 2 or identity.shape != (len(descriptor),):
        raise ValueError("descriptor identity shape drift")
    valid = identity >= 0
    descriptor = F.normalize(descriptor[valid], dim=-1, eps=1e-8)
    identity = identity[valid]
    if len(descriptor) < 3:
        raise ValueError("identity contrastive loss requires valid positive and negative pairs")
    self_mask = torch.eye(len(descriptor), dtype=torch.bool, device=descriptor.device)
    positive = identity[:, None].eq(identity[None, :]) & ~self_mask
    anchors = positive.any(dim=1)
    if not bool(anchors.any()) or len(torch.unique(identity)) < 2:
        raise ValueError("identity contrastive loss requires valid positive and negative pairs")
    logits = descriptor @ descriptor.T / 0.1
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    denominator = torch.exp(logits).masked_fill(self_mask, 0.0).sum(dim=1).clamp_min(1e-12)
    log_probability = logits - torch.log(denominator[:, None])
    return -((log_probability * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1))[anchors].mean()


def axis_anchored_event_relation_core_loss(
    outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Masked event, relation and geometry objective for every causal row."""
    event_target = targets["event_index"].long()
    event_weight = targets.get("event_class_weight")
    if event_weight is not None:
        event_weight = event_weight.to(device=outputs["event_logits"].device, dtype=outputs["event_logits"].dtype)
        if event_weight.shape != (EVENT_CLASSES,) or not bool(torch.isfinite(event_weight).all()) or bool((event_weight <= 0).any()):
            raise ValueError("event class weight must contain five positive finite values")
    event = F.cross_entropy(outputs["event_logits"], event_target, weight=event_weight)
    relation_target = targets["relation_index"].long()
    relation = _balanced_relation_loss(
        outputs["relation_logits_sequence"], relation_target,
        targets.get("relation_positive_rate"),
    )

    if "branch_presence_mask" in targets:
        current_present = targets["branch_presence_mask"].bool()
    else:
        current_present = targets["branch_identity"].long() >= 0
    width_valid = targets["branch_width_valid_mask"].bool()
    if width_valid.shape != current_present.shape or bool((width_valid & ~current_present).any()):
        raise ValueError("branch width mask must be a subset of current branches")
    residual = targets["branch_heading_residual_deg"].to(outputs["event_logits"].dtype)
    width = targets["branch_opening_width_m"].to(outputs["event_logits"].dtype)
    profile = targets["branch_vertical_profile_m"].to(outputs["event_logits"].dtype)
    errors = torch.cat((
        ((outputs["branch_heading_residual_deg"] - residual) / HEADING_RESIDUAL_LIMIT_DEG)[..., None],
        ((torch.log1p(outputs["branch_opening_width_m"]) - torch.log1p(width.clamp_min(0))) / torch.log1p(width.new_tensor(60.0)))[..., None],
        (outputs["branch_vertical_profile_m"] - profile) / PROFILE_SCALE_M,
    ), dim=-1)
    valid = torch.cat((
        current_present[..., None], width_valid[..., None],
        current_present[..., None].expand(-1, -1, PROFILE_DIM),
    ), dim=-1)
    uncertainty = outputs["branch_geometry_uncertainty"]
    if uncertainty.shape != errors.shape or not bool(valid.any(dim=(0, 1)).all()):
        raise ValueError("branch geometry mask/uncertainty contract drift")
    element = F.smooth_l1_loss(errors / uncertainty, torch.zeros_like(errors), reduction="none") + torch.log(uncertainty)
    branch_geometry = torch.stack([element[..., i][valid[..., i]].mean() for i in range(6)]).mean()

    target_axis = F.normalize(targets["local_axis"].to(outputs["local_axis"].dtype), dim=-1)
    axis = (1.0 - (outputs["local_axis"] * target_axis).sum(dim=-1)).mean()
    geometry_target = targets["geometry"].to(outputs["event_logits"].dtype)
    geometry_valid = targets["geometry_valid_mask"].bool()
    prediction = torch.stack((
        outputs["width_m"] / METRIC_DISTANCE_SCALE_M,
        outputs["height_m"] / METRIC_DISTANCE_SCALE_M,
        outputs["slope_deg"] / SLOPE_SCALE_DEG,
        outputs["curvature_per_m"] / CURVATURE_SCALE_PER_M,
    ), dim=-1)
    normalized_target = torch.stack((
        geometry_target[:, 0] / METRIC_DISTANCE_SCALE_M,
        geometry_target[:, 1] / METRIC_DISTANCE_SCALE_M,
        geometry_target[:, 2] / SLOPE_SCALE_DEG,
        geometry_target[:, 3] / CURVATURE_SCALE_PER_M,
    ), dim=-1)
    if geometry_valid.shape != prediction.shape or not bool(geometry_valid.any(dim=0).all()):
        raise ValueError("global geometry target/mask contract drift")
    global_element = F.smooth_l1_loss(prediction, normalized_target, reduction="none")
    global_geometry = torch.stack([global_element[:, i][geometry_valid[:, i]].mean() for i in range(4)]).mean()

    total = event + relation + branch_geometry + axis + global_geometry
    return {
        "total": total, "event": event, "relation": relation,
        "branch_geometry": branch_geometry, "axis": axis,
        "global_geometry": global_geometry,
    }


def axis_anchored_descriptor_loss(
    outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Teacher-identity contrastive objective for identity-balanced batches."""
    place_association = _identity_contrastive(
        outputs["place_descriptor"], targets["association_identity"].long()
    )
    if "branch_presence_mask" in targets:
        branch_mask = targets["branch_presence_mask"].bool()
    else:
        relation_target = targets["relation_index"].long()
        branch_mask = targets["branch_identity"].long() >= 0
    branch_association = _identity_contrastive(
        outputs["branch_descriptor"][branch_mask], targets["branch_identity"].long()[branch_mask]
    )
    total = place_association + branch_association
    return {
        "total": total, "place_association": place_association,
        "branch_association": branch_association,
    }


def axis_anchored_event_relation_loss(
    outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Combined readiness loss; training may schedule core and identity batches."""
    core = axis_anchored_event_relation_core_loss(outputs, targets)
    descriptor = axis_anchored_descriptor_loss(outputs, targets)
    return {
        **{name: value for name, value in core.items() if name != "total"},
        **{name: value for name, value in descriptor.items() if name != "total"},
        "total": core["total"] + descriptor["total"],
    }


def axis_anchored_event_relation_input_contract() -> dict[str, object]:
    return {
        "forward_parameters": tuple(inspect.signature(AxisAnchoredEventRelationNet.forward).parameters)[1:],
        "student_shape": (HISTORY, 2, ELEVATION_ROWS, AZIMUTH_COLUMNS),
        "event_classes": ("corridor", "junction", "terminal", "turn", "geometry_transition"),
        "relation_classes": RELATION_NAMES,
        "axis_anchor": "sensor-forward route motion; bearing bin 0",
        "memory": "four adjacent past-to-current relations only",
        "node_rule": "stable learned structural event with uncertainty refusal",
        "edge_rule": "physical traversal only",
        "teacher_only_loss_fields": ("association_identity", "branch_identity"),
        "forbidden_forward_inputs": (
            "pose", "world_id", "traversal_id", "TNG_identity", "association_identity",
            "exit_identity", "future_frame", "graph_state", "C09", "C10", "M-TARE",
        ),
    }


def parameter_count() -> int:
    return sum(parameter.numel() for parameter in AxisAnchoredEventRelationNet().parameters())


__all__ = [
    "AxisAnchoredEventRelationNet", "BEARING_BINS", "RELATION_NAMES",
    "axis_anchored_descriptor_loss", "axis_anchored_event_relation_core_loss",
    "axis_anchored_event_relation_input_contract", "axis_anchored_event_relation_loss",
    "causal_branch_union_probability", "parameter_count", "reverse_axis_field",
]
