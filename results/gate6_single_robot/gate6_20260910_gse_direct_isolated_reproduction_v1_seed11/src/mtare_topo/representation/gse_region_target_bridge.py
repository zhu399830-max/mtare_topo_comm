"""Loss-only, geometry-only transport of partial structural supervision.

This is NOT an association acceptance test: even a unique, very poor axis
match can transfer a training target. All 32 predictions remain forward
inputs. No identities, relations, teacher centers or configurable physical
distance thresholds enter the axis assignment. Numerical ambiguity loses
supervision explicitly rather than becoming a negative label.
"""
from dataclasses import dataclass
from math import fsum

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from .gse_region_queries import RegionTargets


@dataclass(frozen=True)
class DirectionAlignment:
    teacher_direction: torch.Tensor  # B,64 long; -1 means UNKNOWN
    valid: torch.Tensor              # B,64 bool
    teacher_active: torch.Tensor     # B,32 bool; loss provenance only
    reasons: tuple                   # B tuples of 32 slot status strings
    ledger: tuple                    # per-observation counts, not accuracy


def align_axis_directions(predicted_axes, teacher_axes, teacher_mask):
    """Detached Euclidean/Hungarian matching, including global tie rejection.

    Cost is the mean Euclidean distance of the three corresponding axis
    control points, minimized over endpoint reversal. Unlike coordinate L1,
    it preserves noisy geometric correspondence under a common rotation.
    This new loss bridge does not change any historical geometry scorer.

    For every selected edge, exclude it and solve again. An edge transfers
    labels only if every optimum needs it AND its orientation is unique.
    Unselected rows are labelled UNSELECTED_OR_ALTERNATIVE, not certified
    nonmatches: the solver's arbitrary representative is never supervision.
    Tolerance is fixed float64 rounding allowance (32 eps times maximum point
    norm, times cardinality for summed objectives), not a match threshold.
    """
    values = predicted_axes, teacher_axes, teacher_mask
    if not all(isinstance(x, torch.Tensor) for x in values):
        raise ValueError("axes and teacher mask must be tensors")
    if (predicted_axes.ndim != 4 or predicted_axes.shape[1:] != (32, 3, 3)
            or predicted_axes.shape[0] < 1 or teacher_axes.shape != predicted_axes.shape
            or teacher_mask.shape != predicted_axes.shape[:2] or teacher_mask.dtype != torch.bool):
        raise ValueError("exact B,32,3,3 axes and B,32 boolean teacher mask required")
    if (predicted_axes.dtype not in (torch.float32, torch.float64)
            or teacher_axes.dtype != predicted_axes.dtype
            or any(x.device != predicted_axes.device for x in values)):
        raise ValueError("common float32/64 axes and common device required")
    if not torch.isfinite(predicted_axes).all() or not torch.isfinite(teacher_axes[teacher_mask]).all():
        raise ValueError("nonfinite prediction or active teacher axis")
    device = predicted_axes.device
    mapping = torch.full((len(predicted_axes), 64), -1, dtype=torch.long, device=device)
    all_reasons, ledger = [], []
    for batch in range(len(predicted_axes)):
        active = torch.where(teacher_mask[batch])[0].detach().cpu().numpy()
        statuses = ["UNSELECTED_OR_ALTERNATIVE"] * 32
        counts = {"active_teacher_slots": len(active), "selected_pairs": len(active),
                  "unselected_prediction_slots": 32 - len(active),
                  "ambiguous_assignment_pairs": 0, "ambiguous_orientation_pairs": 0,
                  "unique_direction_pairs": 0, "transferred_directions": 0}
        if len(active):
            pred = predicted_axes[batch].detach().double().cpu().numpy()
            target = teacher_axes[batch, torch.as_tensor(active, device=device)].detach().double().cpu().numpy()
            with np.errstate(over="ignore", invalid="ignore"):
                direct = np.linalg.norm(pred[:, None] - target[None], axis=-1).mean(axis=-1)
                reverse = np.linalg.norm(pred[:, None] - target[None, :, ::-1], axis=-1).mean(axis=-1)
                scale = max(1., float(np.linalg.norm(pred, axis=-1).max()),
                            float(np.linalg.norm(target, axis=-1).max()))
            if not np.isfinite(scale):
                raise ValueError("nonfinite axis-point norm rounding scale")
            cost = np.minimum(direct, reverse)
            if not np.isfinite(cost).all():
                raise ValueError("nonfinite Euclidean assignment cost")
            rows, cols = linear_sum_assignment(cost)
            optimum = fsum(float(cost[r, c]) for r, c in zip(rows, cols))
            if not np.isfinite(optimum):
                raise ValueError("nonfinite assignment objective")
            orientation_tolerance = 32 * np.finfo(np.float64).eps * scale
            objective_tolerance = orientation_tolerance * len(active)
            for row, col in zip(rows, cols):
                excluded = cost.copy()
                excluded[row, col] = np.inf
                try:
                    alternative_rows, alternative_cols = linear_sum_assignment(excluded)
                    alternative = fsum(float(excluded[r, c]) for r, c in zip(alternative_rows, alternative_cols))
                except ValueError:  # no complete assignment after edge exclusion
                    alternative = float("inf")
                if alternative < optimum - objective_tolerance:
                    raise RuntimeError("assignment solver objective inconsistent")
                if alternative <= optimum + objective_tolerance:
                    statuses[row] = "AMBIGUOUS_GLOBAL_ASSIGNMENT"
                    counts["ambiguous_assignment_pairs"] += 1
                elif abs(float(direct[row, col] - reverse[row, col])) <= orientation_tolerance:
                    statuses[row] = "AMBIGUOUS_AXIS_ORIENTATION"
                    counts["ambiguous_orientation_pairs"] += 1
                else:
                    flip = reverse[row, col] < direct[row, col]
                    indices = [2 * int(active[col]) + (1 - e if flip else e) for e in range(2)]
                    mapping[batch, 2 * row:2 * row + 2] = torch.tensor(indices, device=device)
                    statuses[row] = "UNIQUE_GEOMETRY_MATCH_NOT_CONFIDENCE"
                    counts["unique_direction_pairs"] += 1
                    counts["transferred_directions"] += 2
        all_reasons.append(tuple(statuses))
        ledger.append(counts)
    return DirectionAlignment(mapping, mapping >= 0, teacher_mask.detach().clone(),
                              tuple(all_reasons), tuple(ledger))


@dataclass(frozen=True)
class PartialTargetBridge:
    targets: RegionTargets
    ledger: tuple


def partial_region_targets(teacher_rows, alignment, *, dtype):
    """Build padded RegionTargets from supported-construction output rows.

    Unknown events use -1, never a corridor label. Teacher-only identity
    fields are ignored, including for tie handling. Invalid padded members
    remain false, and unmatched predictions receive no negative labels.
    """
    if not isinstance(alignment, DirectionAlignment) or dtype not in (torch.float32, torch.float64):
        raise ValueError("DirectionAlignment and float32/64 dtype required")
    mapping, valid, active = alignment.teacher_direction, alignment.valid, alignment.teacher_active
    if (mapping.ndim != 2 or mapping.shape[1] != 64 or mapping.dtype != torch.long
            or valid.shape != mapping.shape or valid.dtype != torch.bool
            or active.shape != (len(mapping), 32) or active.dtype != torch.bool
            or valid.device != mapping.device or active.device != mapping.device
            or not torch.equal(valid, mapping >= 0) or (mapping < -1).any() or (mapping >= 64).any()):
        raise ValueError("invalid directional alignment")
    if len(teacher_rows) != len(mapping):
        raise ValueError("teacher observation population differs")
    for row, indices in enumerate(mapping):
        known = indices[valid[row]]
        if len(known.unique()) != len(known) or not active[row, known // 2].all():
            raise ValueError("direction mapping must be injective into active teacher slots")
    if any(type(row.get("observation_label_complete")) is not bool
           or row["observation_label_complete"] for row in teacher_rows):
        raise ValueError("only explicitly incomplete supported labels are accepted")
    if any(not isinstance(row.get("regions"), list) for row in teacher_rows):
        raise ValueError("regions must be lists")
    b, m, device = len(mapping), max((len(row["regions"]) for row in teacher_rows), default=0), mapping.device
    centers = torch.zeros((b, m, 3), dtype=dtype, device=device)
    center_valid = torch.zeros((b, m), dtype=torch.bool, device=device)
    events = torch.full((b, m), -1, dtype=torch.long, device=device)
    event_valid = torch.zeros_like(center_valid)
    members = torch.zeros((b, m, 64), dtype=dtype, device=device)
    member_valid = torch.zeros((b, m, 64), dtype=torch.bool, device=device)
    output_ledger = []
    for row, observation in enumerate(teacher_rows):
        counts = dict(regions=len(observation["regions"]), center_labels=0, event_labels=0,
                      original_member_positive=0, original_member_negative=0,
                      transferred_member_positive=0, transferred_member_negative=0,
                      unknown_correspondence_member_positive=0, unknown_correspondence_member_negative=0)
        for region, source in enumerate(observation["regions"]):
            cv, ev = source["center_valid"], source["event_valid"]
            if type(cv) is not bool or type(ev) is not bool:
                raise ValueError("center/event validity must be boolean")
            center = torch.as_tensor(source["center_current_sensor_m"], dtype=dtype, device=device)
            if center.shape != (3,) or not torch.isfinite(center).all():
                raise ValueError("finite three-coordinate teacher center required")
            centers[row, region] = center
            center_valid[row, region], event_valid[row, region] = cv, ev
            event = source["event_target"]
            if ev:
                if event not in ("corridor", "junction"):
                    raise ValueError("supported teacher has no verified terminal event")
                events[row, region] = ("corridor", "junction").index(event)
            elif event is not None:
                raise ValueError("unknown event must have null target")
            labels, mask = source["directional_member_target"], source["directional_member_valid"]
            if (len(labels) != 64 or len(mask) != 64 or any(type(v) is not bool for v in mask)
                    or any(type(v) is not int or v not in (0, 1) for v in labels)):
                raise ValueError("64 binary labels and boolean validity entries required")
            label = torch.tensor(labels, dtype=dtype, device=device)
            known = torch.tensor(mask, dtype=torch.bool, device=device)
            if (known & ~active[row].repeat_interleave(2)).any():
                raise ValueError("teacher direction label references inactive fragment")
            selected = torch.where(valid[row])[0]
            mapped = mapping[row, selected]
            member_valid[row, region, selected] = known[mapped]
            members[row, region, selected] = torch.where(known[mapped], label[mapped], 0.)
            counts["center_labels"] += cv
            counts["event_labels"] += ev
            for name, value in (("positive", 1), ("negative", 0)):
                original = int((known & (label == value)).sum())
                transferred = int((member_valid[row, region] & (members[row, region] == value)).sum())
                counts["original_member_" + name] += original
                counts["transferred_member_" + name] += transferred
                counts["unknown_correspondence_member_" + name] += original - transferred
        output_ledger.append(counts)
    targets = RegionTargets(centers, center_valid, events, event_valid, members, member_valid,
                            torch.zeros(b, dtype=torch.bool, device=device))
    return PartialTargetBridge(targets, tuple(output_ledger))
