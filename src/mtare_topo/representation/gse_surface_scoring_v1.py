"""Independent, partial-label perception scoring; no files or threshold choice.

Every32/64 query enters center-only assignment before any presence readout.
Training matches are never consumed. Global geometric ties make the relevant
main detector score null; conservative count bounds are diagnostics only.
Incomplete/unobserved background is not negative. Position errors on all
uniquely matched queries are oracle-candidate diagnostics, not detection.

Untrained reliability heads are preserved and used only for explicitly named
claim diagnostics. They do not choose matches or silently gate main scores.
No parent confidence interval, paired significance or calibration is claimed.
"""
from dataclasses import dataclass, fields, asdict
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from .gse_surface_losses_v1 import SurfaceLossTargets, _validate
from .gse_surface_relation_model_v1 import SurfaceRelationPrediction


@dataclass(frozen=True)
class SurfaceScoringConfigV1:
    anchor_presence_threshold: float
    opening_presence_threshold: float
    anchor_position_tolerance_m: float
    opening_position_tolerance_m: float
    membership_threshold: float
    reachability_confidence_threshold: float
    membership_validity_threshold: float    # Uncalibrated claim diagnostic only.
    dimension_evidence_threshold: float     # Uncalibrated claim diagnostic only.
    opening_support_threshold: float        # Uncalibrated claim diagnostic only.

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("explicit finite non-bool scoring configuration required")
            if field.name.endswith("tolerance_m"):
                if value <= 0:
                    raise ValueError("explicit positive position tolerance required")
            elif not 0 <= value <= 1:
                raise ValueError("explicit probability threshold outside[0,1]")


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _ratio(a, b):
    return float(a / b) if b else None


def _f1(tp, fp, fn):
    return _ratio(2 * tp, 2 * tp + fp + fn)


def _errors(values):
    return {"count": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "maximum": max(values) if values else None}


def _reach_class(probabilities, threshold):
    maximum = float(probabilities.max()); winners = np.flatnonzero(probabilities == maximum)
    return int(winners[0]) if len(winners) == 1 and maximum >= threshold else 3


def _unique_center_assignment(predicted, centers, valid):
    """Separate evaluator implementation, independent of loss._assign."""
    active = np.flatnonzero(valid)
    mapping = np.full(len(valid), -1, dtype=np.int64)
    distances = np.full(len(valid), np.nan)
    ambiguous = np.zeros(len(valid), dtype=bool)
    if len(active) > len(predicted):
        raise OverflowError("scoreable target population exceeds query capacity; no truncation")
    if not len(active):
        return mapping, distances, ambiguous
    with np.errstate(over="ignore", invalid="ignore"):
        cost = np.linalg.norm(centers[active, None].astype(np.float64) - predicted[None].astype(np.float64), axis=-1)
    if not np.isfinite(cost).all():
        raise ValueError("geometric scoring distance overflow")
    rows, columns = linear_sum_assignment(cost)
    optimum = float(cost[rows, columns].sum())
    tolerance = 64 * np.finfo(np.float64).eps * max(1., optimum, len(active) * float(cost.max()))
    if not math.isfinite(optimum) or not math.isfinite(tolerance):
        raise ValueError("geometric assignment objective overflow")
    for row, column in zip(rows, columns):
        forbidden = cost.copy(); forbidden[row, column] = np.inf
        try:
            ar, ac = linear_sum_assignment(forbidden)
            alternative = float(forbidden[ar, ac].sum())
        except ValueError:
            alternative = math.inf
        target = active[row]
        if alternative <= optimum + tolerance:
            ambiguous[target] = True
        else:
            mapping[target] = column; distances[target] = cost[row, column]
    return mapping, distances, ambiguous


def _detection(prediction, target, prefix, config):
    positions = _numpy(getattr(prediction, prefix + "_position_m"))
    centers = _numpy(getattr(target, prefix + "_position_m"))
    valid = _numpy(getattr(target, prefix + "_valid"))
    probability = _numpy(getattr(prediction, prefix + "_presence_logits").sigmoid())
    declared = probability >= getattr(config, prefix + "_presence_threshold")
    declared &= _numpy(prediction.observation_supported)[:, None]
    complete = _numpy(getattr(target, prefix + "_region_complete"))
    region_center, region_radius = _numpy(target.score_region_center_m), _numpy(target.score_region_radius_m)
    mappings = np.full(valid.shape, -1, dtype=np.int64); detected = np.zeros(valid.shape, dtype=bool)
    counts = dict(raw_queries=positions.shape[0] * positions.shape[1], target_population=int(valid.sum()),
        declared_queries=int(declared.sum()), unique_targets=0, ambiguous_targets=0,
        matched_low_presence=0, matched_outside_position_tolerance=0,
        confirmed_tp=0, complete_region_uncredited_declared=0, unknown_background_declared=0,
        complete_region_background_rejected=0, unmatched_queries=0)
    rows, all_distance, detected_distance = [], [], []
    for b in range(len(positions)):
        mapping, distance, ambiguity = _unique_center_assignment(positions[b], centers[b], valid[b])
        mappings[b] = mapping
        known = np.flatnonzero(valid[b]); unique = mapping >= 0
        detected[b, unique] = declared[b, mapping[unique]] & (distance[unique] <= getattr(config, prefix + "_position_tolerance_m"))
        credited_queries = set(mapping[detected[b]].tolist())
        bound_queries = set(mapping[unique].tolist())
        with np.errstate(over="ignore", invalid="ignore"):
            region_distances = np.linalg.norm(positions[b].astype(np.float64) - region_center[b].astype(np.float64), axis=-1)
        if not np.isfinite(region_distances).all():
            raise ValueError("scoring-region distance overflow")
        inside = region_distances <= region_radius[b]
        uncredited = np.asarray([q not in credited_queries for q in range(len(positions[b]))])
        scoreable_background = inside & complete[b]
        counts["unique_targets"] += int(unique.sum()); counts["ambiguous_targets"] += int(ambiguity.sum())
        counts["confirmed_tp"] += len(credited_queries)
        counts["complete_region_uncredited_declared"] += int((declared[b] & uncredited & scoreable_background).sum())
        counts["unknown_background_declared"] += int((declared[b] & uncredited & ~scoreable_background).sum())
        counts["complete_region_background_rejected"] += int(sum(not declared[b, q] and scoreable_background[q]
            and q not in bound_queries for q in range(len(positions[b]))))
        counts["unmatched_queries"] += len(positions[b]) - len(bound_queries)
        comparisons = []
        for t in known:
            q = int(mapping[t]); d = float(distance[t]) if q >= 0 else None
            comparisons.append({"target_index": int(t), "unique_query_index": q if q >= 0 else None,
                "ambiguous": bool(ambiguity[t]), "distance_m": d,
                "presence_probability": float(probability[b, q]) if q >= 0 else None,
                "declared": bool(declared[b, q]) if q >= 0 else None, "detected": bool(detected[b, t])})
            if q >= 0:
                all_distance.append(d)
                counts["matched_low_presence"] += int(not declared[b, q])
                counts["matched_outside_position_tolerance"] += int(d > getattr(config, prefix + "_position_tolerance_m"))
                if detected[b, t]: detected_distance.append(d)
        rows.append({"observation_index": b, "targets": comparisons,
            "declared_query_indices": np.flatnonzero(declared[b]).tolist(),
            "unknown_background_declared_queries": np.flatnonzero(declared[b] & uncredited & ~scoreable_background).tolist()})
    tp, fp, fn = counts["confirmed_tp"], counts["complete_region_uncredited_declared"], counts["target_population"] - counts["confirmed_tp"]
    ambiguity = counts["ambiguous_targets"]
    score_valid = ambiguity == 0
    summary = {"counts": counts, "valid": score_valid,
        "scope": "observable_foregrounds_and_explicitly_complete_background_only",
        "precision": _ratio(tp, tp + fp) if score_valid else None,
        "recall": _ratio(tp, tp + fn) if score_valid else None,
        "f1": _f1(tp, fp, fn) if score_valid else None,
        "fn": fn if score_valid else None, "fp": fp if score_valid else None,
        "ambiguity_count_bounds_not_scores": {"tp": [tp, min(tp + ambiguity, counts["declared_queries"])],
            "fn": [max(0, fn - ambiguity), fn], "fp": [max(0, fp - ambiguity), fp]},
        "oracle_unique_candidate_position_error_m": _errors(all_distance),
        "detected_position_error_m": _errors(detected_distance), "rows": rows}
    return summary, mappings, detected


@torch.no_grad()
def score_surface_predictions(prediction: SurfaceRelationPrediction, targets: SurfaceLossTargets,
                              config: SurfaceScoringConfigV1):
    if type(config) is not SurfaceScoringConfigV1:
        raise ValueError("explicit scoring thresholds required; no automatic defaults")
    _validate(prediction, targets)  # Reuse types/masks only; never training loss or assignments.
    anchor, amap, adet = _detection(prediction, targets, "anchor", config)
    opening, omap, odet = _detection(prediction, targets, "opening", config)
    p = {f.name: _numpy(getattr(prediction, f.name)) for f in fields(prediction)}
    t = {f.name: _numpy(getattr(targets, f.name)) for f in fields(targets)}
    member_probability = _numpy(prediction.membership_logits.sigmoid())
    member_claim = _numpy(prediction.membership_validity_logits.sigmoid()) >= config.membership_validity_threshold
    dimension_claim = _numpy(prediction.opening_dimension_evidence_logits.sigmoid()) >= config.dimension_evidence_threshold
    reach = _numpy(prediction.reachability_logits.softmax(-1))
    direction_error, width_error, height_error = [], [], []
    ledger = {"direction_known": int(t["direction_valid"].sum()), "direction_invalid_prediction": 0,
        "width_known": int(t["dimension_valid"][..., 0].sum()), "height_known": int(t["dimension_valid"][..., 1].sum()),
        "dimensions_invalid_prediction": 0,
        "membership_known": int(t["membership_valid"].sum()), "membership_scored": 0,
        "membership_known_positive": int((t["membership"][t["membership_valid"]] == 1).sum()),
        "membership_known_negative": int((t["membership"][t["membership_valid"]] == 0).sum()),
        "membership_tp": 0, "membership_fp": 0, "membership_fn": 0, "membership_tn": 0,
        "reachability_reference_known": int(t["reachability_valid"].sum()), "reachability_reference_scored": 0,
        "unknown_reachability_reference_determinate_claims": 0,
        "explicit_unknown_reachability_determinate_claims": 0,
        "unknown_direction_reference_claims": 0, "unknown_dimension_reference_claims": 0,
        "unknown_membership_reference_claims": 0,
        "declared_opening_queries_without_detected_target": 0,
        "unbound_opening_determinate_reachability_claims": 0,
        "unbound_opening_dimension_claims": 0,
        "unbound_opening_membership_claims": 0}
    confusion = np.zeros((3, 4), dtype=np.int64)  # truthtraversable/blocked/unknown x predicted3/reject
    for b in range(len(omap)):
        for u in np.flatnonzero(t["opening_valid"][b]):
            q = int(omap[b, u])
            if q < 0 or not odet[b, u]:
                continue
            direction = p["opening_direction"][b, q].astype(np.float64)
            norm = float(np.linalg.norm(direction))
            # Same numerical unit-vector contract as SurfaceOpeningV1, not an
            # accuracy acceptance threshold or learned confidence gate.
            direction_supported = bool(p["opening_direction_valid"][b, q] and math.isfinite(norm) and abs(norm - 1.) <= 1e-5)
            if t["direction_valid"][b, u]:
                if direction_supported:
                    cosine = float(np.clip((direction / norm) @ t["opening_direction"][b, u], -1., 1.))
                    direction_error.append(math.degrees(math.acos(cosine)))
                else: ledger["direction_invalid_prediction"] += 1
            else: ledger["unknown_direction_reference_claims"] += int(direction_supported)
            for d, errors in ((0, width_error), (1, height_error)):
                if t["dimension_valid"][b, u, d]:
                    size = float(p["opening_dimensions_m"][b, q, d])
                    if size > 0: errors.append(abs(size - float(t["opening_dimensions_m"][b, u, d])))
                    else: ledger["dimensions_invalid_prediction"] += 1
                else: ledger["unknown_dimension_reference_claims"] += int(dimension_claim[b, q, d])
            selected = _reach_class(reach[b, q], config.reachability_confidence_threshold)
            if t["reachability_valid"][b, u]:
                truth = int(t["reachability_class"][b, u]); confusion[truth, selected] += 1
                ledger["reachability_reference_scored"] += 1
                ledger["explicit_unknown_reachability_determinate_claims"] += int(truth == 2 and selected in (0, 1))
            else: ledger["unknown_reachability_reference_determinate_claims"] += int(selected in (0, 1))
            for a in np.flatnonzero(t["anchor_valid"][b]):
                aq = int(amap[b, a])
                if aq < 0 or not adet[b, a]: continue
                if not t["membership_valid"][b, u, a]:
                    ledger["unknown_membership_reference_claims"] += int(member_claim[b, q, aq])
                    continue
                truth = bool(t["membership"][b, u, a]); predicted = bool(member_probability[b, q, aq] >= config.membership_threshold)
                key = "tp" if truth and predicted else "fn" if truth else "fp" if predicted else "tn"
                ledger["membership_" + key] += 1; ledger["membership_scored"] += 1
        # Also disclose confident hypotheses that have no successfully
        # detected reference. They cannot disappear from unknown accounting
        # merely because the best geometric query was low-presence or distant.
        bound = set(omap[b, odet[b]].tolist())
        for q in opening["rows"][b]["declared_query_indices"]:
            if q in bound: continue
            ledger["declared_opening_queries_without_detected_target"] += 1
            ledger["unbound_opening_determinate_reachability_claims"] += int(_reach_class(
                reach[b, q], config.reachability_confidence_threshold) in (0, 1))
            ledger["unbound_opening_dimension_claims"] += int(dimension_claim[b, q].sum())
            ledger["unbound_opening_membership_claims"] += int(member_claim[b, q].sum())
    tp, fp, fn, tn = (ledger["membership_" + k] for k in ("tp", "fp", "fn", "tn"))
    missing_positive = ledger["membership_known_positive"] - tp - fn
    missing_negative = ledger["membership_known_negative"] - fp - tn
    attributes_valid = anchor["valid"] and opening["valid"]
    attributes = {"valid": attributes_valid, "denominators": ledger,
        "direction_error_deg": _errors(direction_error), "width_error_m": _errors(width_error), "height_error_m": _errors(height_error),
        "coverage": {"direction": _ratio(len(direction_error), ledger["direction_known"]),
            "width": _ratio(len(width_error), ledger["width_known"]), "height": _ratio(len(height_error), ledger["height_known"]),
            "membership": _ratio(ledger["membership_scored"], ledger["membership_known"]),
            "reachability": _ratio(ledger["reachability_reference_scored"], ledger["reachability_reference_known"])},
        "membership_known_only_conditional_f1": _f1(tp, fp, fn) if attributes_valid else None,
        "membership_unscored_positive": missing_positive, "membership_unscored_negative": missing_negative,
        "membership_full_known_population_f1_bounds": [_f1(tp, fp + missing_negative, fn + missing_positive),
            _f1(tp + missing_positive, fp, fn)],
        "reachability_confusion": confusion.tolist(),
        "reachability_truth_order": ["traversable", "blocked", "unknown"],
        "reachability_prediction_order": ["traversable", "blocked", "unknown", "reject"],
        "unknown_claims_are_not_false_positive_labels": True,
        "attributes_are_conditional_on_independently_detected_centers": True}
    return {"schema_version": "gse_surface_perception_score_v1", "valid": attributes_valid, "config": asdict(config),
        "anchor": anchor, "opening": opening, "attributes": attributes,
        "raw_predictions": {name: value.tolist() for name, value in p.items()},
        "raw_probabilities": {"anchor_presence": _numpy(prediction.anchor_presence_logits.sigmoid()).tolist(),
            "opening_presence": _numpy(prediction.opening_presence_logits.sigmoid()).tolist(),
            "membership": member_probability.tolist(), "reachability": reach.tolist()},
        "uncalibrated_reliability_diagnostics": {"declared_opening_support_queries": int((
            _numpy(prediction.opening_support_logits.sigmoid()) >= config.opening_support_threshold).sum()),
            "used_for_main_detection_or_matching": False},
        "parent_paired_statistics_or_confidence_intervals_computed": False,
        "scientific_gate_pass": False}
