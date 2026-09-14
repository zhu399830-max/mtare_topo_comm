"""Geometry-bound partial region loss, separate from historical joint loss.

Only center coordinates determine Hungarian assignment and its uniqueness.
Semantic targets never rank candidates. This is a loss interface, not a
deployment association or detector-confidence acceptance rule.
"""
from math import fsum

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from .gse_region_queries import RegionPrediction, RegionTargets


def _validate(prediction, target):
    if not isinstance(prediction, RegionPrediction) or not isinstance(target, RegionTargets):
        raise ValueError("typed RegionPrediction and RegionTargets required")
    q = prediction.query_supported
    if q.ndim != 2 or q.dtype != torch.bool or min(q.shape) < 1:
        raise ValueError("nonempty B,N boolean prediction support required")
    b, n = q.shape
    device, dtype = prediction.centers_m.device, prediction.centers_m.dtype
    if dtype not in (torch.float32, torch.float64) or q.device != device:
        raise ValueError("common float32/64 prediction device required")
    for name, shape in (("centers_m", (b, n, 3)), ("event_logits", (b, n, 3)),
            ("presence_logits", (b, n)), ("membership_logits", (b, n, n)), ("uncertainty", (b, n))):
        x = getattr(prediction, name)
        if x.shape != shape or x.device != device or x.dtype != dtype or not torch.isfinite(x).all():
            raise ValueError("invalid finite prediction " + name)
    support = prediction.member_supported
    if (support.shape != (b, n, n) or support.dtype != torch.bool or support.device != device
            or not torch.equal(support, q[:, :, None] & q[:, None])):
        raise ValueError("member support must be query support outer product")
    if target.centers_m.ndim != 3 or target.centers_m.shape[0] != b or target.centers_m.shape[-1] != 3:
        raise ValueError("target centers must have B,M,3 shape")
    m = target.centers_m.shape[1]
    for name, shape, expected in (("centers_m", (b, m, 3), dtype), ("center_valid", (b, m), torch.bool),
            ("events", (b, m), torch.long), ("event_valid", (b, m), torch.bool),
            ("members", (b, m, n), dtype), ("member_valid", (b, m, n), torch.bool),
            ("label_complete", (b,), torch.bool)):
        x = getattr(target, name)
        if x.shape != shape or x.dtype != expected or x.device != device:
            raise ValueError("target shape/dtype/device drift: " + name)
    if target.label_complete.any():
        raise ValueError("only incomplete partial labels are accepted")
    if ((target.event_valid | target.member_valid.any(-1)) & ~target.center_valid).any():
        raise ValueError("known semantic/member target requires known center")
    if (not torch.isfinite(target.centers_m[target.center_valid]).all()
            or ((target.events[target.event_valid] < 0) | (target.events[target.event_valid] > 2)).any()
            or not torch.isfinite(target.members[target.member_valid]).all()
            or ((target.members[target.member_valid] < 0) | (target.members[target.member_valid] > 1)).any()):
        raise ValueError("invalid known target")
    return b, m, n


def _unique_center_assignment(prediction, target):
    """Original evaluator's float64 geometry matching, without event scoring."""
    pc = prediction.centers_m.detach().double().cpu().numpy()
    tc = target.centers_m.detach().double().cpu().numpy()
    supported = prediction.query_supported.detach().cpu().numpy()
    known = target.center_valid.detach().cpu().numpy()
    unique = np.full(known.shape, -1, dtype=np.int64)
    records = []
    counts = dict(valid_queries=0, invalid_queries=0, valid_centers=0, selected_pairs=0,
                  unique_pairs=0, ambiguous_pairs=0, unselected_queries=0, unselected_centers=0)
    for row in range(len(pc)):
        queries, centers = np.flatnonzero(supported[row]), np.flatnonzero(known[row])
        selected, ambiguities = 0, 0
        if len(queries) and len(centers):
            with np.errstate(over="ignore", invalid="ignore"):
                cost = np.linalg.norm(pc[row, queries][:, None] - tc[row, centers][None], axis=-1)
                scale = max(1., float(np.linalg.norm(pc[row, queries], axis=-1).max()),
                            float(np.linalg.norm(tc[row, centers], axis=-1).max()))
            if not np.isfinite(cost).all() or not np.isfinite(scale):
                raise ValueError("nonfinite center geometry or norm scale")
            ri, ci = linear_sum_assignment(cost)
            optimum = fsum(float(cost[r, c]) for r, c in zip(ri, ci))
            if not np.isfinite(optimum):
                raise ValueError("nonfinite center objective")
            tolerance = 32 * np.finfo(np.float64).eps * scale * len(ri)
            for r, c in zip(ri, ci):
                excluded = cost.copy(); excluded[r, c] = np.inf
                try:
                    ar, ac = linear_sum_assignment(excluded)
                    alternative = fsum(float(excluded[x, y]) for x, y in zip(ar, ac))
                except ValueError:
                    alternative = float("inf")
                if alternative < optimum - tolerance:
                    raise RuntimeError("inconsistent center assignment objective")
                ambiguous = alternative <= optimum + tolerance
                q, t = int(queries[r]), int(centers[c])
                if not ambiguous:
                    unique[row, t] = q
                selected += 1; ambiguities += int(ambiguous)
                # Ambiguous representative pairs are diagnostics only; they
                # must never be consumed as the actual semantic supervision.
                records.append({"row": row, "query": q, "target": t,
                                "distance_m": float(cost[r, c]), "ambiguous": bool(ambiguous)})
        values = dict(valid_queries=len(queries), invalid_queries=len(pc[row]) - len(queries),
            valid_centers=len(centers), selected_pairs=selected, unique_pairs=selected - ambiguities,
            ambiguous_pairs=ambiguities, unselected_queries=len(queries) - selected,
            unselected_centers=len(centers) - selected)
        for key, value in values.items():
            counts[key] += value
    return unique, records, counts


def geometry_bound_region_losses(prediction, target):
    """Five original equally weighted tasks, with geometry-only unique binding.

Fixed batch denominators equal the historical partial loss under this stricter
known-center contract. Labels with ambiguous/unselected geometry stay in those
denominators and are counted, not silently renormalized away. A wholly unknown
or ambiguous batch returns has_supervision=False, never a fit/PASS decision.
"""
    b, m, n = _validate(prediction, target)
    mapping, geometry_matches, geometry_counts = _unique_center_assignment(prediction, target)
    usable_members = target.member_valid & prediction.query_supported[:, None]
    raw_denominators = {"center": int(target.center_valid.sum()), "event": int(target.event_valid.sum()),
        "uncertainty": int(target.event_valid.sum()), "membership": int(usable_members.sum()),
        "presence": sum(min(int(prediction.query_supported[r].sum()), int(target.center_valid[r].sum())) for r in range(b))}
    denominators = {key: max(1, value) for key, value in raw_denominators.items()}
    counts = dict.fromkeys(("targets", "matched", "unmatched_targets", "unsupported_member_targets",
        "unconfirmed_presence_targets", "center", "event", "membership", "presence_positive", "presence_negative"), 0)
    counts["targets"] = int(target.center_valid.sum())
    losses = {key: [] for key in ("center", "event", "membership", "presence", "uncertainty")}
    matches = []
    for row in range(b):
        for t in np.flatnonzero(mapping[row] >= 0):
            q, t = int(mapping[row, t]), int(t)
            matches.append((row, q, t))
            counts["matched"] += 1
            losses["center"].append(torch.linalg.vector_norm(prediction.centers_m[row, q] - target.centers_m[row, t]) / 50)
            counts["center"] += 1
            if target.event_valid[row, t]:
                losses["event"].append(F.cross_entropy(prediction.event_logits[row, q][None], target.events[row, t][None]))
                error = (prediction.event_logits[row, q].detach().argmax() != target.events[row, t]).to(prediction.centers_m.dtype)
                losses["uncertainty"].append((prediction.uncertainty[row, q] - error).square())
                counts["event"] += 1
            valid = usable_members[row, t]
            if valid.any():
                losses["membership"].extend(F.binary_cross_entropy_with_logits(prediction.membership_logits[row, q, valid], target.members[row, t, valid], reduction="none").unbind())
                counts["membership"] += int(valid.sum())
            losses["presence"].append(F.softplus(-prediction.presence_logits[row, q]))
            counts["presence_positive"] += 1
    counts["unmatched_targets"] = counts["targets"] - counts["matched"]
    # Do not sum an entire finite-but-large unused tensor before multiplying
    # by zero: the sum can overflow even though no labels consume that field.
    zero = sum(x.reshape(-1)[:1].sum() * 0 for x in (prediction.centers_m, prediction.event_logits,
        prediction.membership_logits, prediction.presence_logits, prediction.uncertainty))
    result = {key: torch.stack(value).sum() / denominators[key] if value else zero for key, value in losses.items()}
    supervision_counts = {"known_centers": raw_denominators["center"], "known_events": raw_denominators["event"],
        "known_member_labels": int(target.member_valid.sum()), "usable_member_labels": raw_denominators["membership"],
        "unsupported_member_labels": int((target.member_valid & ~prediction.query_supported[:, None]).sum()),
        "unbound_center_labels": raw_denominators["center"] - counts["center"],
        "unbound_event_labels": raw_denominators["event"] - counts["event"],
        "unbound_usable_member_labels": raw_denominators["membership"] - counts["membership"]}
    return {**result, "total": sum(result.values()), "counts": counts, "matches": matches,
        "geometry_matches": geometry_matches, "geometry_counts": geometry_counts,
        "supervision_counts": supervision_counts, "denominators": denominators,
        "unique_center_query": torch.as_tensor(mapping, device=prediction.centers_m.device),
        "has_supervision": bool(matches)}
