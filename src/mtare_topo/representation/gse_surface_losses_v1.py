"""Masked surface-structure supervision with geometry-only unique assignment.

Targets are loss-only. Nothing here produces labels, authorizes training or
calibrates uncertainty. In particular unknown masks are never negative labels
for confidence/validity heads. Positions and dimensions use the fixed10m
length unit; all task weights are1 and every task reports its own denominator.
"""
from dataclasses import dataclass, fields

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from .gse_surface_relation_model_v1 import SurfaceRelationPrediction, ANCHORS, OPENINGS

LENGTH_UNIT_M = 10.
LOSS_NAMES = ("anchor_presence", "anchor_position", "opening_presence", "opening_position",
              "opening_direction", "opening_dimensions", "reachability", "membership")
UNSUPERVISED_HEADS = ("anchor_uncertainty_m", "opening_dimension_evidence_logits",
                      "opening_support_logits", "membership_validity_logits")
SUPERVISED_OUTPUTS = ("anchor_position_m", "anchor_presence_logits", "opening_position_m",
                      "opening_presence_logits", "opening_direction", "opening_dimensions_m",
                      "reachability_logits", "membership_logits")


@dataclass(frozen=True)
class SurfaceLossTargets:
    anchor_position_m: torch.Tensor          # B,T,3; values under false masks may be unknown
    anchor_valid: torch.Tensor               # B,T; observable existence + position known
    opening_position_m: torch.Tensor         # B,U,3
    opening_valid: torch.Tensor              # B,U
    opening_direction: torch.Tensor          # B,U,3; directed unit vector where valid
    direction_valid: torch.Tensor            # B,U
    opening_dimensions_m: torch.Tensor       # B,U,2; positive width/height where valid
    dimension_valid: torch.Tensor            # B,U,2
    reachability_class: torch.Tensor         # B,U integer;0 traversable/1 blocked/2 unknown
    reachability_valid: torch.Tensor         # B,U; explicit known class, NOT implicit all-labels
    physical_reference_valid: torch.Tensor   # B,U; necessary for any root-reach supervision
    membership: torch.Tensor                # B,U,T; independent binary relations
    membership_valid: torch.Tensor           # B,U,T
    score_region_center_m: torch.Tensor      # B,3; complete sphere region for background only
    score_region_radius_m: torch.Tensor      # B; >0, <=10m
    anchor_region_complete: torch.Tensor     # B bool; unmatched inside region can be negative
    opening_region_complete: torch.Tensor    # B bool


@dataclass(frozen=True)
class SurfaceLossResult:
    total: torch.Tensor
    terms: dict
    denominators: dict
    assignments: dict                        # target-index -> query-index, -1 invalid target
    has_supervision: bool
    unsupervised_heads: tuple = UNSUPERVISED_HEADS
    reliability_status: str = "UNTRAINED_UNCALIBRATED"


def _validate(prediction, target):
    if type(prediction) is not SurfaceRelationPrediction or type(target) is not SurfaceLossTargets:
        raise ValueError("explicit prediction and fully masked target types required")
    pos = prediction.anchor_position_m
    if not torch.is_tensor(pos) or pos.ndim != 3 or pos.shape[1:] != (ANCHORS, 3) or pos.shape[0] < 1:
        raise ValueError("fixed32 anchor predictions required")
    b, device, dtype = len(pos), pos.device, pos.dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("float32/64 loss tensors required")
    shapes = dict(anchor_position_m=(b,32,3), anchor_presence_logits=(b,32), anchor_uncertainty_m=(b,32,3),
        opening_position_m=(b,64,3), opening_presence_logits=(b,64), opening_direction=(b,64,3),
        opening_direction_valid=(b,64), opening_dimensions_m=(b,64,2), opening_dimension_evidence_logits=(b,64,2),
        opening_support_logits=(b,64), reachability_logits=(b,64,3), membership_logits=(b,64,32),
        membership_validity_logits=(b,64,32), observation_supported=(b,))
    for field in fields(prediction):
        value = getattr(prediction, field.name); expected_dtype = torch.bool if field.name in ("observation_supported", "opening_direction_valid") else dtype
        if not torch.is_tensor(value) or value.shape != shapes[field.name] or value.dtype != expected_dtype or value.device != device:
            raise ValueError("prediction shape/dtype/device mismatch:" + field.name)
        if expected_dtype != torch.bool and not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite prediction:" + field.name)
    if target.anchor_valid.ndim != 2 or target.opening_valid.ndim != 2:
        raise ValueError("batched explicit target validity required")
    t, u = target.anchor_valid.shape[1], target.opening_valid.shape[1]
    target_shapes = dict(anchor_position_m=(b,t,3), anchor_valid=(b,t), opening_position_m=(b,u,3), opening_valid=(b,u),
        opening_direction=(b,u,3), direction_valid=(b,u), opening_dimensions_m=(b,u,2), dimension_valid=(b,u,2),
        reachability_class=(b,u), reachability_valid=(b,u), physical_reference_valid=(b,u), membership=(b,u,t),
        membership_valid=(b,u,t), score_region_center_m=(b,3), score_region_radius_m=(b,), anchor_region_complete=(b,), opening_region_complete=(b,))
    masks = {"anchor_valid", "opening_valid", "direction_valid", "dimension_valid", "reachability_valid",
             "physical_reference_valid", "membership_valid", "anchor_region_complete", "opening_region_complete"}
    for field in fields(target):
        value = getattr(target, field.name)
        expected_dtype = torch.bool if field.name in masks else torch.long if field.name == "reachability_class" else dtype
        if not torch.is_tensor(value) or value.shape != target_shapes[field.name] or value.dtype != expected_dtype or value.device != device:
            raise ValueError("target shape/dtype/device mismatch:" + field.name)
        if value.requires_grad:
            raise ValueError("teacher targets must be detached")
    for values, mask in ((target.anchor_position_m, target.anchor_valid), (target.opening_position_m, target.opening_valid),
                         (target.opening_direction, target.direction_valid), (target.opening_dimensions_m, target.dimension_valid),
                         (target.membership, target.membership_valid)):
        if not bool(torch.isfinite(values[mask]).all()):
            raise ValueError("nonfinite known target")
    if bool((target.anchor_valid & ~prediction.observation_supported[:,None]).any() or (target.opening_valid & ~prediction.observation_supported[:,None]).any()):
        raise ValueError("known structure targets without observation support")
    if bool((target.direction_valid & ~target.opening_valid).any() or (target.dimension_valid & ~target.opening_valid[...,None]).any()
            or (target.reachability_valid & ~target.opening_valid).any()
            or (target.membership_valid & ~(target.opening_valid[...,None] & target.anchor_valid[:,None])).any()):
        raise ValueError("known attributes require observable geometrically located targets")
    if bool((target.reachability_valid & ~target.physical_reference_valid).any()):
        raise ValueError("root reach supervision requires valid physical reference")
    classes = target.reachability_class[target.reachability_valid]
    if bool(((classes < 0) | (classes > 2)).any()):
        raise ValueError("three-state reachability class required")
    members = target.membership[target.membership_valid]
    if bool(((members != 0) & (members != 1)).any()):
        raise ValueError("known memberships must be independent binary labels")
    direction_norm = torch.linalg.vector_norm(target.opening_direction[target.direction_valid], dim=-1)
    if not torch.allclose(direction_norm, torch.ones_like(direction_norm), atol=64*torch.finfo(dtype).eps, rtol=0):
        raise ValueError("valid target direction must be a directed unit vector")
    if bool((target.opening_dimensions_m[target.dimension_valid] <= 0).any()):
        raise ValueError("positive known opening dimensions required")
    if not bool(torch.isfinite(target.score_region_center_m).all() and torch.isfinite(target.score_region_radius_m).all()) or bool(((target.score_region_radius_m <= 0) | (target.score_region_radius_m > LENGTH_UNIT_M)).any()):
        raise ValueError("explicit finite scoring sphere radius in(0,10]m required")


def _assign(predicted_positions, target_positions, valid):
    """Reject globally ambiguous assignments; confidence never enters cost."""
    predicted = predicted_positions.detach().cpu().double().numpy()
    positions = target_positions.detach().cpu().double().numpy()
    known = valid.detach().cpu().numpy()
    assignment = np.full(known.shape, -1, dtype=np.int64)
    for batch in range(len(predicted)):
        target_ids = np.flatnonzero(known[batch]); centers = positions[batch, target_ids]
        if len(centers) > len(predicted[batch]):
            raise OverflowError("known target count exceeds query capacity; never truncate")
        if not len(centers):
            continue
        if len(np.unique(centers, axis=0)) != len(centers):
            raise ValueError("duplicate known GT centers make assignment ambiguous")
        cost = np.linalg.norm(centers[:,None] - predicted[batch,None], axis=-1)
        if not np.isfinite(cost).all():
            raise ValueError("nonfinite geometric assignment distance")
        rows, columns = linear_sum_assignment(cost)
        optimum = float(cost[rows, columns].sum())
        tolerance = 64*np.finfo(np.float64).eps * max(1., optimum, len(centers)*float(cost.max()))
        for row, column in zip(rows, columns):
            alternative = cost.copy(); alternative[row, column] = np.inf
            try:
                alt_rows, alt_columns = linear_sum_assignment(alternative)
                second = float(alternative[alt_rows, alt_columns].sum())
            except ValueError:
                second = np.inf
            if second <= optimum + tolerance:
                raise ValueError("ambiguous geometry-only global assignment; no favorable slot tie-break")
        assignment[batch, target_ids[rows]] = columns
    return torch.tensor(assignment, dtype=torch.long, device=predicted_positions.device)


def surface_relation_losses(prediction, targets):
    """Eight unit-weight tasks, unknown-safe, full-empty differentiable zero.

    Unmatched negatives use a detached predicted center inside the explicitly
    complete scoring sphere. Incomplete/outside regions are unknown, not FP
    training labels. Validity/uncertainty output heads need their own future
    justified targets; this loss does not invent calibration supervision.
    """
    _validate(prediction, targets)
    # A standalone differentiable zero keeps empty-batch backward legal without
    # attaching absent tasks to model outputs. Zero-valued model gradients would
    # still activate AdamW decay/momentum on wholly unlabelled task heads.
    # Shared features may change from other supervised tasks; this is not a
    # promise that unknown predictions remain numerically frozen.
    zero = torch.zeros((), dtype=prediction.anchor_position_m.dtype,
                       device=prediction.anchor_position_m.device, requires_grad=True)
    numerators = {name: zero for name in LOSS_NAMES}; denominators = {name: 0 for name in LOSS_NAMES}
    anchor_map = _assign(prediction.anchor_position_m, targets.anchor_position_m, targets.anchor_valid)
    opening_map = _assign(prediction.opening_position_m, targets.opening_position_m, targets.opening_valid)

    def add(name, values):
        if not values.numel():
            return  # An empty indexed tensor still creates a backward edge.
        numerators[name] = numerators[name] + values.sum()
        denominators[name] += values.numel()

    for batch in range(len(prediction.anchor_position_m)):
        for prefix, mapping, positions, ground_positions, complete in (
                ("anchor", anchor_map, prediction.anchor_position_m, targets.anchor_position_m, targets.anchor_region_complete),
                ("opening", opening_map, prediction.opening_position_m, targets.opening_position_m, targets.opening_region_complete)):
            known = torch.nonzero(mapping[batch] >= 0, as_tuple=False).flatten(); query = mapping[batch,known]
            logits = getattr(prediction, prefix + "_presence_logits")[batch]
            if len(known):
                add(prefix + "_presence", F.binary_cross_entropy_with_logits(logits[query], torch.ones_like(logits[query]), reduction="none"))
                add(prefix + "_position", torch.linalg.vector_norm(positions[batch,query] - ground_positions[batch,known], dim=-1) / LENGTH_UNIT_M)
            if bool(complete[batch] and prediction.observation_supported[batch]):
                background = torch.linalg.vector_norm(positions[batch].detach() - targets.score_region_center_m[batch], dim=-1) <= targets.score_region_radius_m[batch]
                background[query] = False
                add(prefix + "_presence", F.binary_cross_entropy_with_logits(logits[background], torch.zeros_like(logits[background]), reduction="none"))
        for target_opening in torch.nonzero(opening_map[batch] >= 0, as_tuple=False).flatten().tolist():
            query = opening_map[batch,target_opening]
            if bool(targets.direction_valid[batch,target_opening]):
                cosine = (prediction.opening_direction[batch,query] * targets.opening_direction[batch,target_opening]).sum().clamp(-1.,1.)
                add("opening_direction", ((1.-cosine)/2.).reshape(1))
            mask = targets.dimension_valid[batch,target_opening]
            add("opening_dimensions", (prediction.opening_dimensions_m[batch,query,mask] - targets.opening_dimensions_m[batch,target_opening,mask]).abs()/LENGTH_UNIT_M)
            if bool(targets.reachability_valid[batch,target_opening]):
                add("reachability", F.cross_entropy(prediction.reachability_logits[batch,query][None], targets.reachability_class[batch,target_opening][None], reduction="none"))
            members = torch.nonzero(targets.membership_valid[batch,target_opening], as_tuple=False).flatten()
            queries = anchor_map[batch,members]
            add("membership", F.binary_cross_entropy_with_logits(prediction.membership_logits[batch,query,queries],
                targets.membership[batch,target_opening,members], reduction="none"))
    terms = {name: numerators[name] / max(1, denominators[name]) for name in LOSS_NAMES}
    return SurfaceLossResult(sum(terms.values()), terms, denominators,
                             {"anchor": anchor_map, "opening": opening_map}, any(denominators.values()))


__all__ = ["SurfaceLossTargets", "SurfaceLossResult", "surface_relation_losses", "LOSS_NAMES", "LENGTH_UNIT_M", "UNSUPERVISED_HEADS"]
