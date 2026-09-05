"""Shared-frame geometric composition; no world/place identity inputs.

Geometry is yaw-invariant about the gravity-aligned sensor Z axis, not
independently canonicalized per endpoint. Relative layout survives pooling.
This module does not perform global place recognition or create graph edges.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CompositionInput:
    positions_m: torch.Tensor       # B,N,3 in the same current sensor frame
    tangents: torch.Tensor          # B,N,3 outward unit directions
    half_axes_m: torch.Tensor       # B,N,2; width/height stay gravity anchored
    shape_exponent: torch.Tensor   # B,N
    confidence: torch.Tensor       # B,N
    uncertainty: torch.Tensor      # B,N
    valid: torch.Tensor            # B,N deployable prediction mask, not GT

    def validate(self):
        if self.positions_m.ndim != 3 or self.positions_m.shape[-1] != 3:
            raise ValueError("positions must be B,N,3")
        b, n, _ = self.positions_m.shape
        if b == 0 or not 1 <= n <= 64 or self.valid.shape != (b, n) or self.valid.dtype != torch.bool:
            raise ValueError("invalid composition population/mask")
        for name, tail in (("tangents", (3,)), ("half_axes_m", (2,)),
                           ("shape_exponent", ()), ("confidence", ()), ("uncertainty", ())):
            value = getattr(self, name)
            if value.shape != (b, n, *tail) or not torch.isfinite(value).all():
                raise ValueError(f"invalid {name}")
        if not torch.isfinite(self.positions_m).all():
            raise ValueError("nonfinite positions")
        if ((self.half_axes_m <= 0).any() or (self.shape_exponent <= 0).any()
                or ((self.confidence < 0) | (self.confidence > 1)).any()
                or ((self.uncertainty < 0) | (self.uncertainty > 1)).any()):
            raise ValueError("invalid size/probability")
        norms = torch.linalg.vector_norm(self.tangents, dim=-1)
        if ((norms[self.valid] - 1).abs() > 1e-4).any():
            raise ValueError("valid tangents must have unit norm")


def shared_frame_features(value: CompositionInput):
    """Return B,N,8 unary, B,N,N,8 relational features and pair mask.

    Distances use fixed physical scale 50 m, never validation statistics.
    Signed vertical separation and oriented tangent angles distinguish stacked
    and bent passages. Only a *common* yaw is invariant.
    """
    value.validate()
    p, t = value.positions_m, value.tangents
    unary = torch.cat((
        torch.linalg.vector_norm(p[..., :2], dim=-1, keepdim=True) / 50,
        p[..., 2:3] / 50, torch.log1p(value.half_axes_m),
        value.shape_exponent[..., None], t[..., 2:3],
        value.confidence[..., None], value.uncertainty[..., None],
    ), dim=-1)
    delta = p[:, None, :, :] - p[:, :, None, :]
    ti, tj = t[:, :, None, :], t[:, None, :, :]
    shape = delta.shape[:-1]
    pair = torch.stack((
        torch.linalg.vector_norm(delta, dim=-1) / 50,
        delta[..., 2] / 50, (ti * tj).sum(-1),
        ti[..., 0] * tj[..., 1] - ti[..., 1] * tj[..., 0],
        (delta * ti).sum(-1) / 50, (delta * tj).sum(-1) / 50,
        ti[..., 2].expand(shape), tj[..., 2].expand(shape),
    ), dim=-1)
    mask = value.valid[:, :, None] & value.valid[:, None, :]
    mask = mask & ~torch.eye(p.shape[1], dtype=torch.bool, device=p.device)[None]
    return unary.masked_fill(~value.valid[..., None], 0), pair.masked_fill(~mask[..., None], 0), mask


@dataclass(frozen=True)
class CompositionPrediction:
    event_logits: torch.Tensor       # corridor, junction, terminal
    port_membership_logits: torch.Tensor
    uncertainty: torch.Tensor
    supported: torch.Tensor
    port_supported: torch.Tensor


class GeometricCompositionHead(nn.Module):
    """Small geometry-only baseline head with pairwise message aggregation.

    Validity is explicit. All-unknown inputs are refused, never silently made
    into a terminal. Permuting input endpoints permutes port outputs only.
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.pair_encoder = nn.Sequential(nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.endpoint_encoder = nn.Sequential(nn.Linear(hidden + 8, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.event_head = nn.Linear(hidden, 3)
        self.port_head = nn.Linear(hidden, 1)
        self.uncertainty_head = nn.Linear(hidden, 1)

    def forward(self, value: CompositionInput) -> CompositionPrediction:
        unary, pair, mask = shared_frame_features(value)
        messages = self.pair_encoder(pair).masked_fill(~mask[..., None], 0)
        messages = messages.sum(2) / mask.sum(2).clamp_min(1)[..., None]
        encoded = self.endpoint_encoder(torch.cat((unary, messages), -1))
        weights = value.confidence * value.valid
        supported = weights.sum(1) > 0
        pooled = (encoded * weights[..., None]).sum(1) / weights.sum(1).clamp_min(1e-8)[:, None]
        return CompositionPrediction(
            self.event_head(pooled).masked_fill(~supported[:, None], 0),
            self.port_head(encoded).squeeze(-1).masked_fill(~value.valid, -30),
            torch.where(supported, torch.sigmoid(self.uncertainty_head(pooled).squeeze(-1)), 1.0),
            supported,
            value.valid & supported[:, None],
        )


def masked_structure_losses(prediction, event_target, event_valid, port_target, port_valid):
    """Unknown targets contribute zero, not a fabricated negative label.

    Event validity and port visibility are teacher-only loss arguments. They
    must never be substituted for CompositionInput.valid at deployment.
    """
    b, n = prediction.port_membership_logits.shape
    if (event_target.shape != (b,) or event_valid.shape != (b,)
            or port_target.shape != (b, n) or port_valid.shape != (b, n)
            or event_valid.dtype != torch.bool or port_valid.dtype != torch.bool):
        raise ValueError("target/mask shape mismatch")
    ev = event_valid & prediction.supported
    pv = port_valid & prediction.port_supported
    if ((event_target[ev] < 0) | (event_target[ev] > 2)).any():
        raise ValueError("known event target outside corridor/junction/terminal")
    if not torch.isfinite(port_target[pv]).all() or ((port_target[pv] < 0) | (port_target[pv] > 1)).any():
        raise ValueError("known port target must be a probability")
    event = (F.cross_entropy(prediction.event_logits[ev], event_target[ev]) if ev.any()
             else prediction.event_logits.sum() * 0)
    port = (F.binary_cross_entropy_with_logits(prediction.port_membership_logits[pv], port_target[pv])
            if pv.any() else prediction.port_membership_logits.sum() * 0)
    # Brier supervision for per-observation event error; this needs later
    # held-out calibration and must not be called a calibrated guarantee.
    error = (prediction.event_logits[ev].detach().argmax(-1) != event_target[ev]).to(prediction.uncertainty.dtype)
    uncertainty = ((prediction.uncertainty[ev] - error).square().mean() if ev.any()
                   else prediction.uncertainty.sum() * 0)
    return {"event": event, "port": port, "uncertainty": uncertainty,
            "total": event + port + uncertainty}


def composition_input_from_prediction(prediction, *, existence_threshold, endpoint_threshold):
    """Legacy physical-endpoint-gated bridge, retained for comparison only.

    Do not use this gate to define visibility of entire passages for the new
    node task. A passage can be observed without seeing its construction end.

    Thresholds are explicit caller parameters, not tuned here. Degenerate
    predicted tangents are unknown; they never acquire an invented direction.
    """
    from mtare_topo.representation.primitive_relation_sparse_port_model import outward_endpoint_tangents
    if not 0 <= existence_threshold <= 1 or not 0 <= endpoint_threshold <= 1:
        raise ValueError("invalid frozen prediction thresholds")
    axis = prediction.axis_control_current_sensor_m
    positions = axis[:, :, (0, 2)].reshape(axis.shape[0], -1, 3)
    tangents = outward_endpoint_tangents(axis).reshape_as(positions)
    existence = torch.sigmoid(prediction.existence_logits)[:, :, None].expand(-1, -1, 2)
    evidence = torch.sigmoid(prediction.endpoint_evidence_logits)
    valid = ((existence >= existence_threshold) & (evidence >= endpoint_threshold)).flatten(1)
    valid = valid & (torch.linalg.vector_norm(tangents, dim=-1) >= 1 - 1e-4)
    value = CompositionInput(
        positions, tangents, prediction.endpoint_half_axes_m.flatten(1, 2),
        prediction.endpoint_shape_exponent.flatten(1), (existence * evidence).flatten(1),
        prediction.geometry_uncertainty[:, :, None].expand(-1, -1, 2).flatten(1), valid,
    )
    value.validate()
    return value


def visible_geometry_input_from_prediction(prediction, *, existence_threshold):
    """Node-composition input from visible primitive geometry, not end identity.

    Confidence is predicted primitive existence. The *cropped* geometry and
    its uncertainty are kept even when the old physical-endpoint classifier
    abstains. This does not certify a port, create an edge, or alter the old
    endpoint task's threshold. Degenerate tangents remain invalid.
    """
    from mtare_topo.representation.primitive_relation_sparse_port_model import outward_endpoint_tangents
    if not 0 <= existence_threshold <= 1:
        raise ValueError("invalid frozen existence threshold")
    axis = prediction.axis_control_current_sensor_m
    positions = axis[:, :, (0, 2)].reshape(axis.shape[0], -1, 3)
    tangents = outward_endpoint_tangents(axis).reshape_as(positions)
    existence = torch.sigmoid(prediction.existence_logits)[:, :, None].expand(-1, -1, 2).flatten(1)
    valid = (existence >= existence_threshold) & (torch.linalg.vector_norm(tangents, dim=-1) >= 1 - 1e-4)
    value = CompositionInput(
        positions, tangents, prediction.endpoint_half_axes_m.flatten(1, 2),
        prediction.endpoint_shape_exponent.flatten(1), existence,
        prediction.geometry_uncertainty[:, :, None].expand(-1, -1, 2).flatten(1), valid,
    )
    value.validate()
    return value


def paired_structure_consistency(first, second, port_pairs, common_port_visible, common_event_visible):
    """Teacher-aligned *structure* consistency, never size/shape equality.

    port_pairs is B,K,2: local endpoint slots in the two realizations, and is
    training-only. Unknown pairs may contain -1 and are never indexed. No
    global place identity is a forward input or a contrastive negative.
    """
    b = first.event_logits.shape[0]
    if (port_pairs.ndim != 3 or port_pairs.shape[0] != b or port_pairs.shape[-1] != 2
            or port_pairs.dtype != torch.long or common_port_visible.shape != port_pairs.shape[:2]
            or common_port_visible.dtype != torch.bool or common_event_visible.shape != (b,)
            or common_event_visible.dtype != torch.bool or second.event_logits.shape != (b, 3)):
        raise ValueError("invalid realization correspondence")
    ev = common_event_visible & first.supported & second.supported
    event = ((first.event_logits[ev].softmax(-1) - second.event_logits[ev].softmax(-1)).square().mean()
             if ev.any() else (first.event_logits.sum() + second.event_logits.sum()) * 0)
    rows, cols = torch.where(common_port_visible)
    left, right = port_pairs[rows, cols, 0], port_pairs[rows, cols, 1]
    if ((left < 0) | (left >= first.port_membership_logits.shape[1])
            | (right < 0) | (right >= second.port_membership_logits.shape[1])).any():
        raise ValueError("known port correspondence outside prediction")
    visible = first.port_supported[rows, left] & second.port_supported[rows, right]
    rows, left, right = rows[visible], left[visible], right[visible]
    port = ((first.port_membership_logits[rows, left].sigmoid()
             - second.port_membership_logits[rows, right].sigmoid()).square().mean()
            if len(rows) else (first.port_membership_logits.sum() + second.port_membership_logits.sum()) * 0)
    return {"event": event, "port": port, "total": event + port}
