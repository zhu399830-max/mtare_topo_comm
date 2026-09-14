"""CPU-only, same-snapshot attribution of geometry versus training assignment.

Joint assignment is a GT-label-conditioned representative, with uniqueness
NOT verified. Its semantic numbers explain optimization, never replace the
independent geometry evaluation or constitute scientific qualification.
"""
from dataclasses import fields

import numpy as np
import torch

from mtare_topo.evaluation.gse_partial_structure import (
    evaluate_partial_structure, _events, _binary_f1, _ratio)
from mtare_topo.representation.gse_region_queries import (
    RegionPrediction, RegionTargets, region_set_losses)


def _slice(value, indices):
    return type(value)(**{f.name: getattr(value, f.name)[indices] for f in fields(value)})


def _statistics(values):
    return dict(count=len(values), mean=float(np.mean(values)) if values else None,
                median=float(np.median(values)) if values else None,
                p10=float(np.percentile(values, 10)) if values else None,
                p90=float(np.percentile(values, 90)) if values else None)


def _semantics(prediction, target, row, ti, query, rejection):
    known = target.member_valid[row, ti]
    supported = prediction.query_supported[row]
    usable = known & supported if query is not None else torch.zeros_like(known)
    positives = usable & (target.members[row, ti] == 1)
    negatives = usable & (target.members[row, ti] == 0)
    if query is None:
        probability = torch.zeros_like(target.members[row, ti])
        logits, predicted, tied = None, None, None
    else:
        probability = torch.sigmoid(prediction.membership_logits[row, query])
        logits_tensor = prediction.event_logits[row, query]
        winners = torch.where(logits_tensor == logits_tensor.max())[0]
        logits, tied = logits_tensor.tolist(), len(winners) != 1
        predicted = int(winners[0]) if not tied else None
    positive_values, negative_values = probability[positives].tolist(), probability[negatives].tolist()
    decision = probability >= .5
    counts = dict(tp=int((positives & decision).sum()), fn=int((positives & ~decision).sum()),
                  fp=int((negatives & decision).sum()), tn=int((negatives & ~decision).sum()))
    rejected = {name: int((known & ~usable & (target.members[row, ti] == value)).sum())
                for name, value in (("positive", 1), ("negative", 0))}
    return dict(event_argmax=predicted, event_logits=logits, event_maximum_tied=tied,
                members=counts, member_rejected=rejected,
                member_rejection_reason=rejection if query is None else "unsupported_direction",
                positive_probability=_statistics(positive_values), negative_probability=_statistics(negative_values),
                _positive_values=positive_values, _negative_values=negative_values)


def _summarize(records, bridge, mode):
    confusion = np.zeros((2, 4), dtype=np.int64)
    counts = dict(tp=0, fp=0, fn=0, tn=0)
    rejected = {"direction_correspondence_unknown": dict(positive=0, negative=0)}
    original = dict(positive=0, negative=0)
    supplied = dict(positive=0, negative=0)
    for ledger in bridge:
        for name in ("positive", "negative"):
            original[name] += ledger["original_member_" + name]
            supplied[name] += ledger["transferred_member_" + name]
            rejected["direction_correspondence_unknown"][name] += ledger["unknown_correspondence_member_" + name]
    positive_values, negative_values, distances = [], [], []
    for record in records:
        result = record[mode]
        for key, value in result["members"].items(): counts[key] += value
        reason = result["member_rejection_reason"]
        rejected.setdefault(reason, dict(positive=0, negative=0))
        for name, value in result["member_rejected"].items(): rejected[reason][name] += value
        positive_values.extend(result["_positive_values"])
        negative_values.extend(result["_negative_values"])
        if record[mode + "_distance_m"] is not None: distances.append(record[mode + "_distance_m"])
        if record["event_truth"] is not None:
            outcome = result["event_argmax"]
            confusion[record["event_truth"], 3 if outcome is None else outcome] += 1
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    missing_p = sum(value["positive"] for value in rejected.values())
    missing_n = sum(value["negative"] for value in rejected.values())
    if tp + fn + missing_p != original["positive"] or fp + tn + missing_n != original["negative"]:
        raise RuntimeError("attribution member population does not conserve original labels")
    return dict(events=_events(confusion), center_distance_m=_statistics(distances),
                members={**counts, "original_positive": original["positive"], "original_negative": original["negative"],
                    "supplied_positive": supplied["positive"], "supplied_negative": supplied["negative"],
                    "rejected": rejected, "positive_coverage": _ratio(tp + fn, original["positive"]),
                    "negative_coverage": _ratio(fp + tn, original["negative"]),
                    "coverage": _ratio(tp + fp + fn + tn, sum(original.values())),
                    "known_scored_precision": _ratio(tp, tp + fp), "known_scored_recall": _ratio(tp, tp + fn),
                    "known_scored_f1": _binary_f1(tp, fp, fn),
                    "worst_case_f1": _binary_f1(tp, fp + missing_n, fn + missing_p),
                    "best_case_f1": _binary_f1(tp + missing_p, fp, fn),
                    "positive_probability": _statistics(positive_values), "negative_probability": _statistics(negative_values),
                    "member_threshold": .5, "threshold_calibrated": False,
                    "bound_population": "PRE_BRIDGE_KNOWN_LABELS"})


def _comparison_counts(records):
    both = [r for r in records if r["same_query"] is not None]
    return dict(known_targets=len(records), both_assigned=len(both),
                same_query=sum(r["same_query"] for r in both),
                different_query=sum(not r["same_query"] for r in both),
                incomparable_targets=len(records)-len(both),
                joint_minus_geometry_distance_m=_statistics([r["joint_distance_m"] - r["geometry_distance_m"]
                    for r in both if r["joint_distance_m"] is not None and r["geometry_distance_m"] is not None]))


@torch.no_grad()
def diagnose_assignments(prediction: RegionPrediction, target: RegionTargets, manifest: list,
                         bridge: list, last_epoch_batches: list[list[int]]) -> dict:
    """Recompute frozen joint cost with the exact prior epoch batch grouping.

    All rows occur exactly once. Inputs are only existing predictions and
    targets: this function has no model, weights, file IO, or optimizer API.
    Teacher-only metadata is used solely to group/report outputs. Joint ties
    are retained as solver representatives, never called unique or verified.
    """
    if not isinstance(prediction, RegionPrediction) or not isinstance(target, RegionTargets):
        raise ValueError("RegionPrediction/RegionTargets required")
    if any(getattr(value, f.name).device.type != "cpu" for value in (prediction, target) for f in fields(value)):
        raise ValueError("CPU cached tensors required; no device transfer or inference")
    b = prediction.centers_m.shape[0]
    if (not isinstance(manifest, list) or len(manifest) != b
            or any(not isinstance(r, dict) or type(r.get("task")) is not str or not r["task"] for r in manifest)):
        raise ValueError("one task-bearing manifest row per observation required")
    if (not isinstance(bridge, list) or len(bridge) != b
            or any(not isinstance(r, dict) or not isinstance(r.get("target_transport"), dict) for r in bridge)):
        raise ValueError("one target_transport bridge ledger per observation required")
    if (not isinstance(last_epoch_batches, list) or not last_epoch_batches
            or any(not isinstance(batch, list) or not batch or any(type(i) is not int for i in batch)
                   for batch in last_epoch_batches)):
        raise ValueError("nonempty integer batch lists required")
    if sorted(i for batch in last_epoch_batches for i in batch) != list(range(b)):
        raise ValueError("last epoch must partition all observations exactly once")
    transport = [row["target_transport"] for row in bridge]
    geometry = evaluate_partial_structure(prediction, target, membership_threshold=.5,
                                          manifest=manifest, direction_bridge_ledger=transport)
    joint_queries = torch.full_like(geometry.unique_center_query, -1)
    batches = []
    for indices in last_epoch_batches:
        selection = torch.tensor(indices, dtype=torch.long)
        result = region_set_losses(_slice(prediction, selection), _slice(target, selection))
        for local_row, query, ti in result["matches"]:
            row = indices[local_row]
            if joint_queries[row, ti] != -1: raise RuntimeError("duplicate target joint assignment")
            joint_queries[row, ti] = query
        batches.append(dict(observation_indices=list(indices),
                            loss={key: float(result[key]) for key in
                                  ("total", "center", "event", "membership", "presence", "uncertainty")},
                            counts=dict(result["counts"]),
                            joint_representative_matches=[dict(observation_index=indices[r], query=q, target_index=t)
                                                          for r, q, t in result["matches"]]))
    records = []
    for row in range(b):
        known = target.center_valid[row] | target.event_valid[row] | target.member_valid[row].any(-1)
        for ti in torch.where(known)[0].tolist():
            qg, qj = int(geometry.unique_center_query[row, ti]), int(joint_queries[row, ti])
            qg, qj = qg if qg >= 0 else None, qj if qj >= 0 else None
            center = target.centers_m[row, ti].tolist() if target.center_valid[row, ti] else None
            record = dict(observation_index=row, target_index=ti, manifest=dict(manifest[row]),
                          geometry_query=qg, joint_query=qj, teacher_center_m=center,
                          geometry_center_m=prediction.centers_m[row, qg].tolist() if qg is not None else None,
                          joint_center_m=prediction.centers_m[row, qj].tolist() if qj is not None else None,
                          same_query=qg == qj if qg is not None and qj is not None else None,
                          event_truth=int(target.events[row, ti]) if target.event_valid[row, ti] else None,
                          geometry_center_status=geometry.observations[row]["target_center_status"][ti],
                          joint_assignment_unique_verified=False)
            for mode, query, reason in (("geometry", qg, record["geometry_center_status"]),
                                        ("joint", qj, "joint_unmatched_or_ineligible")):
                record[mode + "_distance_m"] = (float(torch.linalg.vector_norm(
                    prediction.centers_m[row, query].double() - target.centers_m[row, ti].double()))
                    if query is not None and center is not None else None)
                record[mode] = _semantics(prediction, target, row, ti, query, reason)
            records.append(record)
    joint_summary = _summarize(records, transport, "joint")
    joint_summary.update(scope="GT_LABEL_CONDITIONED_REPRESENTATIVE_ASSIGNMENT_ATTRIBUTION_ONLY",
                         assignment_uniqueness_verified=False, scientific_score=False)
    # Independently accumulated per-target geometry semantics must agree with
    # the authoritative evaluator before reporting any attribution contrast.
    geometry_detail = _summarize(records, transport, "geometry")
    if (geometry_detail["events"] != geometry.summary["events"] or any(
            geometry_detail["members"][key] != geometry.summary["members"][key]
            for key in ("tp", "fp", "fn", "tn", "coverage", "worst_case_f1", "best_case_f1"))):
        raise RuntimeError("geometry attribution does not reproduce independent evaluation")
    parents = {}
    for parent in sorted({row["task"] for row in manifest}):
        indices = [i for i, row in enumerate(manifest) if row["task"] == parent]
        selected_records = [record for record in records if record["observation_index"] in indices]
        ledgers = [transport[i] for i in indices]
        parent_joint = _summarize(selected_records, ledgers, "joint")
        parent_joint.update(scope="GT_LABEL_CONDITIONED_REPRESENTATIVE_ASSIGNMENT_ATTRIBUTION_ONLY",
                            assignment_uniqueness_verified=False, scientific_score=False)
        parents[parent] = dict(comparison=_comparison_counts(selected_records),
                               geometry=_summarize(selected_records, ledgers, "geometry"), joint_diagnostic=parent_joint)
    for record in records:
        for mode in ("geometry", "joint"):
            record[mode].pop("_positive_values"); record[mode].pop("_negative_values")
    return dict(scope="SAME_CACHED_SNAPSHOT_ASSIGNMENT_ATTRIBUTION_NO_MODEL_OR_TRAINING",
                observation_count=b, last_epoch_batches=[list(batch) for batch in last_epoch_batches],
                geometry_summary=geometry.summary, geometry_probability_diagnostic=geometry_detail,
                joint_diagnostic=joint_summary, comparison=_comparison_counts(records),
                target_comparisons=records, junction_table=[r for r in records if r["event_truth"] == 1],
                parents=parents, batch_joint_losses=batches,
                joint_assignment_uniqueness_verified=False, replaces_independent_scoring=False,
                member_threshold=.5, threshold_search=False, model_forwards=0, optimizer_steps=0)
