"""Observation-direct and geometry-composition software prototype.

A and B have identical forwards and parameter layouts. B's auxiliary geometry
loss is a TRAINER responsibility; C additionally exposes predicted geometry to
the structure decoder. All branches retain the raw observation path. Context
is supplied by a caller and detached: this module does not load an encoder,
checkpoint, teacher, frame identity, or coordinate transform.

All queries are retained. Support means a nonempty observation, NOT calibrated
detection support. Empty observations are unknown (not labelled no-object).
Widths and heights are predictions, with separate evidence probabilities.
Generic coordinate MLPs and attention are NOT strictly yaw/SE(3) equivariant.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .gse_point_axis_readout import PointAxisReadout
from .primitive_relation_model import MINIMUM_SHAPE_EXPONENT, MAXIMUM_SHAPE_EXPONENT


DIM = 128
PRIMITIVES = 32
STRUCTURES = 32
PORTALS = 64
EVENT_ORDER = ("corridor", "junction", "terminal", "no_object")


@dataclass(frozen=True)
class PrimitiveGeometry:
    axis_control_m: torch.Tensor       # B,32,3,3
    half_axes_m: torch.Tensor          # B,32,2,2 (two end sections)
    exponent: torch.Tensor            # B,32,2
    confidence_probability: torch.Tensor  # B,32
    direction: torch.Tensor           # B,32,3; zero if unresolved
    direction_defined: torch.Tensor   # B,32 bool
    observation_supported: torch.Tensor  # B,32 bool, not detection evidence


@dataclass(frozen=True)
class DualPathPrediction:
    primitives: PrimitiveGeometry
    structure_position_m: torch.Tensor    # B,32,3
    structure_event_logits: torch.Tensor  # B,32,4; EVENT_ORDER
    structure_presence_probability: torch.Tensor  # B,32
    portal_position_m: torch.Tensor       # B,64,3, independent learned queries
    portal_direction: torch.Tensor        # B,64,3
    portal_direction_defined: torch.Tensor # B,64 bool
    portal_dimensions_m: torch.Tensor     # B,64,2 positive predicted width,height
    portal_dimension_evidence_probability: torch.Tensor  # B,64,2
    portal_presence_probability: torch.Tensor  # B,64
    portal_membership_probability: torch.Tensor  # B,64,33; final=unknown
    observation_supported: torch.Tensor   # B bool; false => ALL semantics unknown


def _direction(vector, supported):
    norm = torch.linalg.vector_norm(vector, dim=-1)
    tolerance = 32 * torch.finfo(vector.dtype).eps * vector.abs().amax(-1).clamp_min(1.)
    defined = (norm > tolerance) & supported
    unit = vector / norm.clamp_min(torch.finfo(vector.dtype).tiny)[..., None]
    return torch.where(defined[..., None], unit, 0.), defined


def _decoder():
    layer = nn.TransformerDecoderLayer(DIM, 4, dim_feedforward=4 * DIM,
        dropout=0., activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerDecoder(layer, num_layers=2, norm=nn.LayerNorm(DIM))


class DualPathModelV1(nn.Module):
    """Fixed 128-D/two-layer/four-head model with paired A/B/C parameters.

    Input XYZ and frozen context must describe the SAME already motion-aligned
    points in one current sensor frame. Caller, not this tensor interface,
    must establish that provenance. No point order/primitive order embedding.
    Empty rows use a numerical sentinel only; outputs remain explicitly unknown.
    """
    def __init__(self, path):
        super().__init__()
        if path not in ("A", "B", "C"):
            raise ValueError("path must be A, B or C")
        self.path = path
        self.shared_point_adapter = nn.Sequential(nn.Linear(DIM + 3, DIM),
            nn.GELU(), nn.Linear(DIM, DIM), nn.LayerNorm(DIM))
        self.primitive_queries = nn.Parameter(torch.randn(PRIMITIVES, DIM) / math.sqrt(DIM))
        self.structure_queries = nn.Parameter(torch.randn(STRUCTURES, DIM) / math.sqrt(DIM))
        self.portal_queries = nn.Parameter(torch.randn(PORTALS, DIM) / math.sqrt(DIM))
        self.primitive_decoder = _decoder()
        self.structure_decoder = _decoder()
        self.axis_readout = PointAxisReadout(model_dim=DIM, point_dim=32)
        self.primitive_section = nn.Linear(DIM, 6)
        self.primitive_confidence = nn.Linear(DIM, 1)
        self.geometry_adapter = nn.Sequential(nn.Linear(19, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.structure_position = nn.Linear(DIM, 3)
        self.structure_event = nn.Linear(DIM, 4)
        self.portal_position = nn.Linear(DIM, 3)
        self.portal_orientation = nn.Linear(DIM, 3)
        self.portal_dimensions = nn.Linear(DIM, 2)
        self.portal_dimension_evidence = nn.Linear(DIM, 2)
        self.portal_presence = nn.Linear(DIM, 1)
        self.membership_portal = nn.Linear(DIM, DIM)
        self.membership_structure = nn.Linear(DIM, DIM)
        self.membership_unknown = nn.Linear(DIM, 1)

    def encode_observation(self, points_xyz_m, frozen_point_context, valid):
        if points_xyz_m.ndim != 3 or points_xyz_m.shape[-1] != 3:
            raise ValueError("points must be B,N,3")
        b, n, _ = points_xyz_m.shape
        if b < 1 or n < 1 or frozen_point_context.shape != (b, n, DIM):
            raise ValueError("nonempty B,N,128 context required")
        if valid.shape != (b, n) or valid.dtype != torch.bool:
            raise ValueError("valid must be bool B,N")
        if points_xyz_m.dtype not in (torch.float32, torch.float64):
            raise ValueError("float32/64 coordinates required")
        if (frozen_point_context.dtype != points_xyz_m.dtype or
                any(v.device != points_xyz_m.device for v in (valid, frozen_point_context))):
            raise ValueError("input dtype/device drift")
        points = torch.where(valid[..., None], points_xyz_m, 0.)
        context = torch.where(valid[..., None], frozen_point_context.detach(), 0.)
        if not all(bool(torch.isfinite(v).all()) for v in (points, context)):
            raise ValueError("nonfinite valid observation")
        supported = valid.any(-1)
        safe_valid = valid.clone()
        safe_valid[~supported, 0] = True
        feature = self.shared_point_adapter(torch.cat((points / 50., context), -1))
        feature = torch.where(valid[..., None], feature, 0.)
        centroid = points.sum(1) / valid.sum(1).clamp_min(1)[..., None]
        return points, feature, safe_valid, supported, centroid

    def predict_geometry(self, points, feature, safe_valid, supported):
        b, n, _ = points.shape
        slots = self.primitive_decoder(self.primitive_queries[None].expand(b, -1, -1),
            feature, memory_key_padding_mask=~safe_valid)
        index = torch.arange(n, device=points.device)[None].expand(b, -1)
        readout = self.axis_readout(points, safe_valid, feature, points, index, slots)
        axis = torch.where(supported[:, None, None, None], readout.votes.axis_control_m, 0.)
        raw_sections = self.primitive_section(slots)
        half_axes = F.softplus(raw_sections[..., :4]) + torch.finfo(points.dtype).eps
        exponent = MINIMUM_SHAPE_EXPONENT + (MAXIMUM_SHAPE_EXPONENT - MINIMUM_SHAPE_EXPONENT) * raw_sections[..., 4:].sigmoid()
        support = supported[:, None].expand(-1, PRIMITIVES)
        direction, defined = _direction(axis[:, :, 2] - axis[:, :, 0],
            support & readout.votes.direction_defined)
        confidence = self.primitive_confidence(slots).squeeze(-1).sigmoid() * support
        return PrimitiveGeometry(axis, half_axes.reshape(b, PRIMITIVES, 2, 2),
            exponent, confidence, direction, defined, support)

    def decode_structure(self, feature, safe_valid, supported, centroid, geometry):
        """Internal candidate decoder; geometry is predicted, never target-selected.

        Exposed separately for synthetic intervention tests. A/B intentionally
        do not inspect geometry. C treats primitive slots as an unordered set.
        """
        b = feature.shape[0]
        memory, memory_valid = feature, safe_valid
        if self.path == "C":
            fields = torch.cat((geometry.axis_control_m.flatten(2) / 50., geometry.direction,
                geometry.half_axes_m.flatten(2) / 50., geometry.exponent,
                geometry.confidence_probability[..., None]), -1)
            if fields.shape != (b, PRIMITIVES, 19) or not bool(torch.isfinite(fields).all()):
                raise ValueError("predicted geometry shape/nonfinite drift")
            # No confidence threshold or GT membership: every primitive remains.
            soft_geometry = self.geometry_adapter(fields) * geometry.confidence_probability[..., None]
            memory = torch.cat((memory, soft_geometry), 1)
            memory_valid = torch.cat((memory_valid, torch.ones((b, PRIMITIVES),
                dtype=torch.bool, device=feature.device)), 1)
        queries = torch.cat((self.structure_queries, self.portal_queries), 0)
        decoded = self.structure_decoder(queries[None].expand(b, -1, -1), memory,
            memory_key_padding_mask=~memory_valid)
        structures, portals = decoded[:, :STRUCTURES], decoded[:, STRUCTURES:]
        structure_xyz = centroid[:, None] + 50. * self.structure_position(structures)
        portal_xyz = centroid[:, None] + 50. * self.portal_position(portals)
        portal_support = supported[:, None].expand(-1, PORTALS)
        direction, defined = _direction(self.portal_orientation(portals), portal_support)
        membership_logits = torch.einsum("bpd,bsd->bps", self.membership_portal(portals),
            self.membership_structure(structures)) / math.sqrt(DIM)
        membership = torch.cat((membership_logits, self.membership_unknown(portals)), -1).softmax(-1)
        unknown = torch.zeros_like(membership)
        unknown[..., -1] = 1.
        event_logits = torch.where(supported[:, None, None], self.structure_event(structures), 0.)
        return DualPathPrediction(geometry,
            torch.where(supported[:, None, None], structure_xyz, 0.),
            event_logits,
            (1. - event_logits.softmax(-1)[..., 3]) * supported[:, None],
            torch.where(supported[:, None, None], portal_xyz, 0.), direction, defined,
            F.softplus(self.portal_dimensions(portals)) + torch.finfo(feature.dtype).eps,
            self.portal_dimension_evidence(portals).sigmoid() * supported[:, None, None],
            self.portal_presence(portals).squeeze(-1).sigmoid() * supported[:, None],
            torch.where(supported[:, None, None], membership, unknown), supported)

    def forward(self, points_xyz_m, frozen_point_context, valid):
        points, feature, safe_valid, supported, centroid = self.encode_observation(
            points_xyz_m, frozen_point_context, valid)
        geometry = self.predict_geometry(points, feature, safe_valid, supported)
        result = self.decode_structure(feature, safe_valid, supported, centroid, geometry)
        # Do not let finite inputs with numerical overflow become valid evidence.
        values = list(vars(result).values()) + list(vars(geometry).values())
        if any(torch.is_tensor(v) and v.is_floating_point() and not bool(torch.isfinite(v).all())
               for v in values):
            raise ValueError("nonfinite dual-path prediction")
        return result
