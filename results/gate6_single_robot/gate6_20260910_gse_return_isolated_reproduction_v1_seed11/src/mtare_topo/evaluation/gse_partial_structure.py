"""Geometry-matched partial-label capacity diagnostics, NOT detection metrics.

No training matches are accepted or consulted. Every numerical query remains
a candidate, regardless of presence. Teacher centers select an optimal
candidate assignment, so even zero localization error does not demonstrate
autonomous detection. Missing/ambiguous labels are explicit, never negatives.
"""
from dataclasses import dataclass
from math import fsum

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from mtare_topo.representation.gse_region_queries import RegionPrediction, RegionTargets


@dataclass(frozen=True)
class PartialCapacityEvaluation:
    summary: dict
    observations: tuple
    scored_member_mask: torch.Tensor  # CPU B,M,N, original target/token order
    unique_center_query: torch.Tensor  # CPU B,M; -1 means not uniquely matched


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _binary_f1(tp, fp, fn):
    return _ratio(2 * tp, 2 * tp + fp + fn)


def _events(confusion):
    per_class = {}
    for label, name in enumerate(("corridor", "junction")):
        tp = int(confusion[label, label])
        fp = int(confusion[:, label].sum()) - tp
        fn = int(confusion[label].sum()) - tp
        support = int(confusion[label].sum())
        per_class[name] = dict(support=support, precision=_ratio(tp, tp + fp),
                               recall=_ratio(tp, support), f1=_binary_f1(tp, fp, fn))
    supported = all(value["support"] > 0 for value in per_class.values())
    return dict(true_classes=["corridor", "junction"],
                predicted_classes=["corridor", "junction", "terminal", "reject"],
                confusion=confusion.tolist(), per_class=per_class,
                two_class_macro_f1=sum(v["f1"] for v in per_class.values()) / 2 if supported else None,
                two_class_macro_available=supported,
                known_events=int(confusion.sum()), rejected_events=int(confusion[:, 3].sum()),
                accuracy_diagnostic_only=_ratio(int(confusion[0, 0] + confusion[1, 1]), int(confusion.sum())))


def _distance_summary(values):
    return dict(count=len(values), mean_m=float(np.mean(values)) if values else None,
                median_m=float(np.median(values)) if values else None,
                p90_m=float(np.percentile(values, 90)) if values else None)


def _validate(prediction, target, threshold, manifest, bridge):
    if (isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or threshold != .5):
        raise ValueError("explicit membership_threshold=0.5 required; no calibration/search")
    if not isinstance(prediction, RegionPrediction) or not isinstance(target, RegionTargets):
        raise ValueError("RegionPrediction and RegionTargets required")
    q = prediction.query_supported
    if q.ndim != 2 or q.dtype != torch.bool or q.shape[0] < 1 or q.shape[1] < 1:
        raise ValueError("nonempty B,N boolean query support required")
    b, n = q.shape
    device, dtype = q.device, prediction.centers_m.dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("float32/64 predictions required")
    for name, shape in (("centers_m", (b, n, 3)), ("event_logits", (b, n, 3)),
                        ("presence_logits", (b, n)), ("membership_logits", (b, n, n)),
                        ("uncertainty", (b, n))):
        x = getattr(prediction, name)
        if x.shape != shape or x.dtype != dtype or x.device != device or not torch.isfinite(x).all():
            raise ValueError("invalid prediction " + name)
    support = prediction.member_supported
    if (support.shape != (b, n, n) or support.dtype != torch.bool or support.device != device
            or not torch.equal(support, q[:, :, None] & q[:, None])):
        raise ValueError("member support must be query support outer product")
    if target.centers_m.ndim != 3 or target.centers_m.shape[0] != b:
        raise ValueError("target centers must be B,M,3")
    m = target.centers_m.shape[1]
    for name, shape, expected_dtype in (
            ("centers_m", (b, m, 3), dtype), ("center_valid", (b, m), torch.bool),
            ("events", (b, m), torch.long), ("event_valid", (b, m), torch.bool),
            ("members", (b, m, n), dtype), ("member_valid", (b, m, n), torch.bool),
            ("label_complete", (b,), torch.bool)):
        x = getattr(target, name)
        if x.shape != shape or x.dtype != expected_dtype or x.device != device:
            raise ValueError("invalid target " + name)
    if target.label_complete.any():
        raise ValueError("this evaluator is only for incomplete labels")
    if (not torch.isfinite(target.centers_m[target.center_valid]).all()
            or not ((target.events[target.event_valid] == 0) | (target.events[target.event_valid] == 1)).all()
            or not ((target.members[target.member_valid] == 0) | (target.members[target.member_valid] == 1)).all()):
        raise ValueError("finite centers, supported two-class events and binary known members required")
    if manifest is not None and (len(manifest) != b or any(not isinstance(row, dict) for row in manifest)):
        raise ValueError("manifest must provide one metadata dictionary per observation")
    if bridge is not None and len(bridge) != b:
        raise ValueError("direction bridge ledger population differs")
    return b, m, n


def evaluate_partial_structure(prediction, target, *, membership_threshold,
                               manifest=None, direction_bridge_ledger=None):
    """Score center-selected candidates with fixed, uncalibrated .5 members.

    Optional bridge ledger is PartialTargetBridge.ledger: it restores original
    known-positive/negative counts lost in direction correspondence. Without
    it, bounds cover supplied RegionTargets only, explicitly not pre-bridge
    labels. Geometry ties use fixed float64 roundoff, not a physical gate.
    Manifest is carried as metadata only and never affects assignment.
    """
    b, m, n = _validate(prediction, target, membership_threshold, manifest, direction_bridge_ledger)
    def cpu(x): return x.detach().cpu()
    pc = cpu(prediction.centers_m).double().numpy()
    tc = cpu(target.centers_m).double().numpy()
    supported = cpu(prediction.query_supported)
    center_valid = cpu(target.center_valid)
    events, event_valid = cpu(target.events), cpu(target.event_valid)
    member, member_valid = cpu(target.members), cpu(target.member_valid)
    event_logits, member_logits = cpu(prediction.event_logits), cpu(prediction.membership_logits)
    scored_mask = torch.zeros_like(member_valid)
    unique_query = torch.full((b, m), -1, dtype=torch.long)
    confusion = np.zeros((2, 4), dtype=np.int64)
    counters = dict(tp=0, fp=0, fn=0, tn=0, supplied_positive=0, supplied_negative=0,
                    original_positive=0, original_negative=0)
    reasons = ("direction_correspondence_unknown", "center_missing", "center_unselected_or_alternative",
               "center_ambiguous", "unsupported_direction")
    rejected = {reason: dict(positive=0, negative=0) for reason in reasons}
    observations, all_distances, unique_distances = [], [], []
    geometry = dict(valid_queries=0, invalid_queries=0, valid_centers=0,
                    selected_pairs=0, unique_pairs=0, ambiguous_pairs=0,
                    unselected_queries=0, unselected_centers=0)
    for row in range(b):
        queries = torch.where(supported[row])[0].numpy()
        centers = torch.where(center_valid[row])[0].numpy()
        status = ["center_missing" if not v else "center_unselected_or_alternative" for v in center_valid[row].tolist()]
        assigned = []
        if len(queries) and len(centers):
            with np.errstate(over="ignore", invalid="ignore"):
                cost = np.linalg.norm(pc[row, queries][:, None] - tc[row, centers][None], axis=-1)
                scale = max(1., float(np.linalg.norm(pc[row, queries], axis=-1).max()),
                            float(np.linalg.norm(tc[row, centers], axis=-1).max()))
            if not np.isfinite(cost).all() or not np.isfinite(scale):
                raise ValueError("nonfinite center geometry or norm scale")
            ri, ci = linear_sum_assignment(cost)
            optimum = fsum(float(cost[r, c]) for r, c in zip(ri, ci))
            if not np.isfinite(optimum): raise ValueError("nonfinite center objective")
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
                q, t, distance = int(queries[r]), int(centers[c]), float(cost[r, c])
                status[t] = "center_ambiguous" if ambiguous else "unique"
                if not ambiguous:
                    unique_query[row, t] = q
                    unique_distances.append(distance)
                all_distances.append(distance)
                assigned.append(dict(query=q, target=t, distance_m=distance, ambiguous=bool(ambiguous)))
        row_geometry = dict(valid_queries=len(queries), invalid_queries=n-len(queries), valid_centers=len(centers),
                            selected_pairs=len(assigned), unique_pairs=int((unique_query[row] >= 0).sum()),
                            ambiguous_pairs=sum(p["ambiguous"] for p in assigned),
                            unselected_queries=len(queries)-len(assigned), unselected_centers=len(centers)-len(assigned))
        for key, value in row_geometry.items(): geometry[key] += value
        row_rejected = {reason: dict(positive=0, negative=0) for reason in reasons}
        row_counts = dict(tp=0, fp=0, fn=0, tn=0)
        supplied = {name: int((member_valid[row] & (member[row] == value)).sum())
                    for name, value in (("positive", 1), ("negative", 0))}
        for name in ("positive", "negative"):
            original, lost = supplied[name], 0
            if direction_bridge_ledger is not None:
                ledger = direction_bridge_ledger[row]
                original = ledger["original_member_" + name]
                transferred = ledger["transferred_member_" + name]
                lost = ledger["unknown_correspondence_member_" + name]
                if (any(type(v) is not int or v < 0 for v in (original, transferred, lost))
                        or transferred != supplied[name] or original != transferred + lost):
                    raise ValueError("direction bridge population conservation failed")
            counters["supplied_" + name] += supplied[name]
            counters["original_" + name] += original
            row_rejected["direction_correspondence_unknown"][name] = lost
        row_confusion = np.zeros((2, 4), dtype=np.int64)
        for t in range(m):
            q = int(unique_query[row, t])
            if event_valid[row, t]:
                label = int(events[row, t]); outcome = 3
                if q >= 0:
                    logits = event_logits[row, q]
                    # Exact equal class maxima are a reject, not an arbitrary
                    # class-index preference. No truth label breaks the tie.
                    winners = torch.where(logits == logits.max())[0]
                    if len(winners) == 1: outcome = int(winners[0])
                row_confusion[label, outcome] += 1
            known = member_valid[row, t]
            if q < 0:
                for name, value in (("positive", 1), ("negative", 0)):
                    row_rejected[status[t]][name] += int((known & (member[row, t] == value)).sum())
                continue
            usable = known & supported[row]
            scored_mask[row, t] = usable
            for name, value in (("positive", 1), ("negative", 0)):
                row_rejected["unsupported_direction"][name] += int((known & ~supported[row] & (member[row, t] == value)).sum())
            truth = member[row, t, usable] == 1
            positive = torch.sigmoid(member_logits[row, q, usable]) >= membership_threshold
            row_counts["tp"] += int((positive & truth).sum())
            row_counts["fp"] += int((positive & ~truth).sum())
            row_counts["fn"] += int((~positive & truth).sum())
            row_counts["tn"] += int((~positive & ~truth).sum())
        confusion += row_confusion
        for key, value in row_counts.items(): counters[key] += value
        for reason in reasons:
            for name in ("positive", "negative"): rejected[reason][name] += row_rejected[reason][name]
        observations.append(dict(index=row, manifest=dict(manifest[row]) if manifest is not None else None,
                                 geometry=row_geometry, center_assignment=assigned, target_center_status=status,
                                 event_confusion=row_confusion.tolist(), members=row_counts,
                                 supplied_member_counts=supplied, member_rejected=row_rejected))
    tp, fp, fn, tn = (counters[k] for k in ("tp", "fp", "fn", "tn"))
    missing_p = sum(r["positive"] for r in rejected.values())
    missing_n = sum(r["negative"] for r in rejected.values())
    if tp + fn + missing_p != counters["original_positive"] or fp + tn + missing_n != counters["original_negative"]:
        raise RuntimeError("member evaluation population conservation failed")
    baseline = np.zeros_like(confusion); baseline[:, 0] = confusion.sum(axis=1)
    summary = dict(scope="PARTIAL_GEOMETRY_ORACLE_CANDIDATE_CAPACITY_NOT_DETECTION_OR_GRAPH",
                   observations=b, geometry=geometry,
                   assigned_distance_representative=_distance_summary(all_distances),
                   unique_assigned_distance=_distance_summary(unique_distances),
                   events=_events(confusion), always_corridor_baseline=_events(baseline),
                   members={**counters, "rejected": rejected,
                       "positive_coverage": _ratio(tp + fn, counters["original_positive"]),
                       "negative_coverage": _ratio(fp + tn, counters["original_negative"]),
                       "coverage": _ratio(tp + fp + fn + tn, counters["original_positive"] + counters["original_negative"]),
                       "known_scored_precision": _ratio(tp, tp + fp), "known_scored_recall": _ratio(tp, tp + fn),
                       "known_scored_f1": _binary_f1(tp, fp, fn),
                       "worst_case_f1": _binary_f1(tp, fp + missing_n, fn + missing_p),
                       "best_case_f1": _binary_f1(tp + missing_p, fp, fn),
                       "bound_population": "PRE_BRIDGE_KNOWN_LABELS" if direction_bridge_ledger is not None else "SUPPLIED_REGION_TARGETS_ONLY",
                       "membership_threshold": membership_threshold, "threshold_calibrated": False},
                   unmatched_queries_are_false_positives=False, three_class_gate_pass_claim=False,
                   complete_detection_claim=False)
    return PartialCapacityEvaluation(summary, tuple(observations), scored_mask, unique_query)
