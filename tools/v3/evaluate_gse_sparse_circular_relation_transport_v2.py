#!/usr/bin/env python3
"""Select on C07 and transfer once to C08 for sparse relation transport V2."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
import math
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import train_gse_axis_anchored_event_relation_v1 as base
import train_gse_sparse_circular_relation_transport_v2 as sparse
from mtare_topo.evaluation.gse_metrics import relative_geometry_improvement
from mtare_topo.governance import write_json
from mtare_topo.semantics.range_geometry_baseline import RangeGeometryBaseline


PASS = "PASS_GSE_SPARSE_CIRCULAR_RELATION_TRANSPORT_SELECTION_V2"
FAIL = "FAIL_GSE_SPARSE_CIRCULAR_RELATION_TRANSPORT_SELECTION_V2"
EVENT_BASELINE = {"c07": 0.6622920066, "c08": 0.6337416128}
EVENT_GAIN = 0.05
TOKEN_PRECISION_FLOOR = 0.995
TOKEN_RECALL_FLOOR = 0.50
RELATION_PRECISION_FLOOR = 0.98
RELATION_RECALL_FLOOR = 0.25
ASSOCIATION_PRECISION_FLOOR = 0.98
ASSOCIATION_FALSE_CEILING = 0.01
ASSOCIATION_RECALL_FLOOR = 0.25
STRUCTURAL_PRECISION_FLOOR = 0.98
STRUCTURAL_FALSE_CEILING = 0.01
STRUCTURAL_RECALL_FLOOR = 0.25
LOCALIZATION_TOLERANCE_BINS = 1
PERMUTATIONS_6 = np.asarray(tuple(itertools.permutations(range(6))), dtype=np.int16)


def _normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def _softmax(value: np.ndarray, axis: int = -1) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    value = value - value.max(axis=axis, keepdims=True)
    value = np.exp(value)
    return value / value.sum(axis=axis, keepdims=True)


def _circular_distance(left, right):
    distance = np.abs(np.asarray(left) - np.asarray(right))
    return np.minimum(distance, 180 - distance)


def _circular_nms(logits: np.ndarray) -> np.ndarray:
    flat = np.asarray(logits, dtype=np.float64).reshape(-1, 180).copy()
    selected = np.empty((len(flat), 6), dtype=np.int16)
    rows = np.arange(len(flat))[:, None]
    offsets = np.arange(-4, 5)[None]
    for token in range(6):
        index = flat.argmax(1).astype(np.int16)
        selected[:, token] = index
        suppressed = np.remainder(index[:, None] + offsets, 180)
        flat[rows, suppressed] = -np.inf
    return selected.reshape(*logits.shape[:-1], 6)


def _align_six(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return candidate indices aligned to each reference token."""
    reference = np.asarray(reference, dtype=np.int16)
    candidate = np.asarray(candidate, dtype=np.int16)
    if reference.shape != candidate.shape or reference.shape[-1] != 6:
        raise ValueError("six-token alignment shape drift")
    flat_reference = reference.reshape(-1, 6)
    flat_candidate = candidate.reshape(-1, 6)
    result = np.empty_like(flat_reference)
    target = np.arange(6)
    for start in range(0, len(result), 512):
        stop = min(start + 512, len(result))
        cost = _circular_distance(flat_reference[start:stop, :, None], flat_candidate[start:stop, None, :])
        candidate_cost = cost[
            :, target[None, :, None], PERMUTATIONS_6.T[None, :, :]
        ].sum(axis=2)[:, 0, :]
        result[start:stop] = PERMUTATIONS_6[candidate_cost.argmin(1)]
    return result.reshape(reference.shape)


def _gather_token(value: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    index = mapping
    while index.ndim < value.ndim:
        index = index[..., None]
    return np.take_along_axis(value, index, axis=mapping.ndim - 1)


def _deployed_token_coordinates_bins(ensemble: dict[str, np.ndarray]) -> np.ndarray:
    """Return the continuous deployed token bearing in 2-degree bin units."""
    bearing = np.asarray(ensemble["token_bearing_deg"], dtype=np.float64)
    hard_index = np.asarray(ensemble["token_bin_index"])
    if bearing.shape != hard_index.shape or not np.isfinite(bearing).all():
        raise ValueError("deployed token bearing shape/finite drift")
    return np.remainder(bearing / 2.0, 180.0)


def _circular_mean(values: list[np.ndarray]) -> np.ndarray:
    radians = [np.deg2rad(value.astype(np.float64)) for value in values]
    return np.remainder(np.rad2deg(np.arctan2(np.mean([np.sin(x) for x in radians], axis=0), np.mean([np.cos(x) for x in radians], axis=0))), 360.0)


def _ensemble(predictions: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    proposal = np.mean([p["proposal_logits"].astype(np.float64) for p in predictions], axis=0)
    reference_bin = _circular_nms(proposal)
    mappings = [_align_six(reference_bin, p["token_bin_index"]) for p in predictions]
    token_values = {}
    for name in ("token_descriptor", "token_opening_width_m", "token_vertical_profile_m", "token_geometry_uncertainty"):
        aligned = [_gather_token(p[name].astype(np.float64), mapping) for p, mapping in zip(predictions, mappings)]
        token_values[name] = np.mean(aligned, axis=0)
    token_values["token_descriptor"] = _normalize(token_values["token_descriptor"])
    aligned_bearings = [_gather_token(p["token_bearing_deg"].astype(np.float64)[..., None], mapping)[..., 0] for p, mapping in zip(predictions, mappings)]
    token_values["token_bearing_deg"] = _circular_mean(aligned_bearings)
    aligned_rows = []
    aligned_reveal = []
    for prediction, mapping in zip(predictions, mappings):
        row = prediction["transport_row_probability"].astype(np.float64)
        reveal = prediction["transport_reveal_probability"].astype(np.float64)
        output_row = np.empty_like(row)
        output_reveal = np.empty_like(reveal)
        for step in range(4):
            previous = mapping[:, step]
            current = mapping[:, step + 1]
            batch = np.arange(len(row))[:, None]
            real = row[:, step][batch, previous]
            output_row[:, step, :, :6] = np.take_along_axis(real[:, :, :6], current[:, None, :], axis=2)
            output_row[:, step, :, 6] = real[:, :, 6]
            output_reveal[:, step] = reveal[:, step][batch, current]
        aligned_rows.append(output_row)
        aligned_reveal.append(output_reveal)
    result = {
        "proposal_logits": proposal,
        "token_bin_index": reference_bin,
        "transport_row_probability": np.mean(aligned_rows, axis=0),
        "transport_reveal_probability": np.mean(aligned_reveal, axis=0),
        "event_logits": np.mean([p["event_logits"].astype(np.float64) for p in predictions], axis=0),
        "token_count_probability": np.mean([p["token_count_probability"].astype(np.float64) for p in predictions], axis=0),
        "geometry": np.mean([p["geometry"].astype(np.float64) for p in predictions], axis=0),
        "local_axis": _normalize(np.mean([p["local_axis"].astype(np.float64) for p in predictions], axis=0)),
        "place_descriptor": _normalize(np.mean([p["place_descriptor"].astype(np.float64) for p in predictions], axis=0)),
        "observation_uncertainty": np.mean([p["observation_uncertainty"].astype(np.float64) for p in predictions], axis=0),
        **token_values,
    }
    result["count"] = result["token_count_probability"].argmax(-1)
    return result


def _binary_metrics(score, truth, threshold):
    score = np.asarray(score, dtype=np.float64); truth = np.asarray(truth, dtype=bool); selected = score >= threshold
    tp = int((selected & truth).sum()); fp = int((selected & ~truth).sum()); fn = int((~selected & truth).sum()); tn = int((~selected & ~truth).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12), "false_merge_fraction": fp / max(tp + fp, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accepted": tp + fp, "positives": tp + fn}


def _objective_recall_metrics(score, truth, threshold, objective_positive_total):
    """Score detections while counting every objective positive in recall.

    ``truth`` labels the emitted candidates.  It cannot represent a teacher
    token or relation whose endpoint was never emitted, so its ordinary FN
    count is incomplete.  Precision remains candidate-based, while recall,
    FN, positives and F1 use the independently counted teacher population.
    """
    metrics = _binary_metrics(score, truth, threshold)
    objective_positive_total = int(objective_positive_total)
    if objective_positive_total < metrics["tp"]:
        raise ValueError("objective positive total is smaller than true positives")
    metrics["positives"] = objective_positive_total
    metrics["fn"] = objective_positive_total - metrics["tp"]
    metrics["recall"] = metrics["tp"] / max(objective_positive_total, 1)
    metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / max(
        metrics["precision"] + metrics["recall"], 1e-12
    )
    return metrics


def _select_threshold(score, truth, precision_floor, false_ceiling=None):
    score = np.asarray(score, dtype=np.float64); truth = np.asarray(truth, dtype=bool)
    if not truth.any() or not (~truth).any():
        raise ValueError("threshold population lacks both classes")
    candidates = np.unique(score[truth]); best = None
    for threshold in candidates:
        metrics = _binary_metrics(score, truth, threshold)
        if metrics["precision"] < precision_floor or (false_ceiling is not None and metrics["false_merge_fraction"] > false_ceiling):
            continue
        if best is None or (metrics["recall"], metrics["precision"]) > (best["recall"], best["precision"]):
            best = metrics
    return best


def _event_metrics(logits, truth):
    prediction = logits.argmax(1); matrix = np.zeros((5, 5), dtype=np.int64); np.add.at(matrix, (truth, prediction), 1)
    names = ("corridor", "junction", "terminal", "turn", "geometry_transition"); values = []; per_class = {}
    for index, name in enumerate(names):
        tp = int(matrix[index, index]); p = int(matrix[:, index].sum()); n = int(matrix[index].sum()); precision = tp / p if p else 0; recall = tp / n if n else 0; f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        values.append(f1); per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": n}
    return {"macro_f1": float(np.mean(values)), "accuracy": float(np.trace(matrix) / matrix.sum()), "per_class": per_class, "confusion": matrix.tolist()}


def _load_pairs(path: Path, condition: int):
    result = defaultdict(list)
    with path.open() as stream:
        for line in stream:
            record = json.loads(line); match = re.search(r"_C(\d+)$", record["parent_id"])
            if match and int(match.group(1)) == condition: result[record["parent_id"]].append(record)
    return result


def _match_prediction_to_truth(predicted_bins, predicted_count, truth_identity):
    mapping = np.full(6, -1, dtype=np.int64); distance = np.full(6, np.inf); count = int(np.clip(predicted_count, 0, 6)); true_bins = np.flatnonzero(truth_identity >= 0)
    if not count or not len(true_bins): return mapping, distance
    best = None
    for chosen in itertools.permutations(range(count), min(count, len(true_bins))):
        if len(true_bins) > count: break
        cost = _circular_distance(predicted_bins[list(chosen)], true_bins).sum()
        if best is None or cost < best[0]: best = (cost, chosen)
    if len(true_bins) <= count:
        chosen = best[1]
        for true_index, token in enumerate(chosen):
            mapping[token] = int(truth_identity[true_bins[true_index]]); distance[token] = _circular_distance(predicted_bins[token], true_bins[true_index])
    else:
        # Enumerate true subsets when prediction count is smaller.
        best = None
        for true_subset in itertools.permutations(range(len(true_bins)), count):
            cost = _circular_distance(predicted_bins[:count], true_bins[list(true_subset)]).sum()
            if best is None or cost < best[0]: best = (cost, true_subset)
        for token, true_index in enumerate(best[1]):
            mapping[token] = int(truth_identity[true_bins[true_index]]); distance[token] = _circular_distance(predicted_bins[token], true_bins[true_index])
    return mapping, distance


def _world_metrics(world, ensemble, pairs, baseline):
    token_score = []; token_correct = []; token_positive_total = 0; exact = []
    relation = {
        name: {"score": [], "truth": [], "positive_total": 0}
        for name in ("persistent", "reveal", "withdraw")
    }
    width_error = []; profile_error = []
    mappings = np.full((len(world["event_index"]), 5, 6), -1, dtype=np.int64)
    distances = np.full_like(mappings, np.inf, dtype=np.float64)
    history = world["history_observation_rows"]
    deployed_coordinates = _deployed_token_coordinates_bins(ensemble)
    for row in range(len(world["event_index"])):
        for frame in range(5):
            history_row = history[row, frame]
            if history_row < 0: continue
            mappings[row, frame], distances[row, frame] = _match_prediction_to_truth(deployed_coordinates[row, frame], ensemble["count"][row, frame], world["branch_identity"][history_row])
        current_count = int(ensemble["count"][row, -1]); true_count = int((world["branch_identity"][row] >= 0).sum()); token_positive_total += true_count
        confidence = float(ensemble["token_count_probability"][row, -1, current_count])
        for token in range(current_count):
            score = confidence * float(1 / (1 + np.exp(-ensemble["proposal_logits"][row, -1, ensemble["token_bin_index"][row, -1, token]])))
            correct = mappings[row, -1, token] >= 0 and distances[row, -1, token] <= LOCALIZATION_TOLERANCE_BINS
            token_score.append(score); token_correct.append(correct)
            if correct:
                teacher_bin = np.flatnonzero(world["branch_identity"][row] == mappings[row, -1, token])[0]
                if world["branch_width_valid_mask"][row, teacher_bin]: width_error.append(abs(float(ensemble["token_opening_width_m"][row, -1, token]) - float(world["branch_opening_width_m"][row, teacher_bin])))
                profile_error.extend(np.abs(ensemble["token_vertical_profile_m"][row, -1, token] - world["branch_vertical_profile_m"][row, teacher_bin]).tolist())
        exact.append(current_count == true_count and sum(distances[row, -1, :current_count] <= LOCALIZATION_TOLERANCE_BINS) == true_count)
        for step in range(4):
            if history[row, step] < 0 or history[row, step + 1] < 0: continue
            previous_true = set(int(x) for x in world["branch_identity"][history[row, step]] if x >= 0); current_true = set(int(x) for x in world["branch_identity"][history[row, step + 1]] if x >= 0)
            relation["persistent"]["positive_total"] += len(previous_true & current_true)
            relation["reveal"]["positive_total"] += len(current_true - previous_true)
            relation["withdraw"]["positive_total"] += len(previous_true - current_true)
            previous_count = int(ensemble["count"][row, step]); current_count_step = int(ensemble["count"][row, step + 1])
            for previous in range(previous_count):
                previous_identity = mappings[row, step, previous] if distances[row, step, previous] <= LOCALIZATION_TOLERANCE_BINS else -1
                relation["withdraw"]["score"].append(float(ensemble["transport_row_probability"][row, step, previous, 6])); relation["withdraw"]["truth"].append(previous_identity >= 0 and previous_identity not in current_true)
                for current in range(current_count_step):
                    current_identity = mappings[row, step + 1, current] if distances[row, step + 1, current] <= LOCALIZATION_TOLERANCE_BINS else -1
                    relation["persistent"]["score"].append(float(ensemble["transport_row_probability"][row, step, previous, current])); relation["persistent"]["truth"].append(previous_identity >= 0 and previous_identity == current_identity)
            for current in range(current_count_step):
                current_identity = mappings[row, step + 1, current] if distances[row, step + 1, current] <= LOCALIZATION_TOLERANCE_BINS else -1
                relation["reveal"]["score"].append(float(ensemble["transport_reveal_probability"][row, step, current])); relation["reveal"]["truth"].append(current_identity >= 0 and current_identity not in previous_true)
    geometry_baseline_sum = np.zeros(4); geometry_baseline_count = np.zeros(4, dtype=np.int64)
    for row in range(len(world["event_index"])):
        frame = int(world["references"][row, -1]); estimate = baseline.predict(world["range_m"][frame], world["valid_mask"][frame]); values = np.asarray([estimate[n] for n in ("width_m", "height_m", "slope_deg", "curvature_per_m")]); valid = world["geometry_valid_mask"][row]; geometry_baseline_sum += np.abs(values - world["geometry"][row]) * valid; geometry_baseline_count += valid
    row_by_global = {int(value): row for row, value in enumerate(world["global_sequence_index"])}; association = {"place": [], "branch": [], "combined": [], "truth": []}
    for pair in pairs:
        left = row_by_global[int(pair["anchor_global_sequence_index"])]; right = row_by_global[int(pair["paired_global_sequence_index"])]; place = float(ensemble["place_descriptor"][left] @ ensemble["place_descriptor"][right]); left_count = int(ensemble["count"][left, -1]); right_count = int(ensemble["count"][right, -1]); branch = float((ensemble["token_descriptor"][left, -1, :left_count] @ ensemble["token_descriptor"][right, -1, :right_count].T).max()) if left_count and right_count else -1.0
        association["place"].append(place); association["branch"].append(branch); association["combined"].append(.5 * (place + branch)); association["truth"].append(bool(pair["same_identity"]))
    return {"token_score": token_score, "token_correct": token_correct, "token_positive_total": token_positive_total, "exact": exact, "relation": relation, "width_error": width_error, "profile_error": profile_error, "association": association, "geometry_baseline_sum": geometry_baseline_sum, "geometry_baseline_count": geometry_baseline_count}


def _split(condition, dataset_root, traversals, prediction_roots, pair_manifest):
    parents = sorted(parent for parent in traversals if parent.endswith(f"_C{condition:02d}")); pairs = _load_pairs(pair_manifest, condition); baseline = RangeGeometryBaseline(); collection = defaultdict(list); world_rows = []
    geometry_baseline_sum = np.zeros(4); geometry_baseline_count = np.zeros(4, dtype=np.int64)
    for parent in parents:
        world = sparse._load_world(dataset_root, parent, traversals[parent]); predictions = [dict(np.load(root/f"{parent}.npz")) for root in prediction_roots]
        if any(not np.array_equal(p["global_sequence_index"], world["global_sequence_index"]) for p in predictions): raise RuntimeError(f"prediction join drift: {parent}")
        ensemble = _ensemble(predictions); metrics = _world_metrics(world, ensemble, pairs.get(parent, []), baseline)
        collection["event_logits"].append(ensemble["event_logits"]); collection["event_truth"].append(world["event_index"]); collection["geometry"].append(ensemble["geometry"]); collection["geometry_truth"].append(world["geometry"]); collection["geometry_valid"].append(world["geometry_valid_mask"]); collection["axis"].append(ensemble["local_axis"]); collection["axis_truth"].append(world["local_axis"]); collection["uncertainty"].append(ensemble["observation_uncertainty"]); collection["token_score"].extend(metrics["token_score"]); collection["token_correct"].extend(metrics["token_correct"]); collection["token_positive_total"].append(metrics["token_positive_total"]); collection["exact"].extend(metrics["exact"]); collection["width_error"].extend(metrics["width_error"]); collection["profile_error"].extend(metrics["profile_error"])
        for name in metrics["relation"]: collection[f"relation_{name}_score"].extend(metrics["relation"][name]["score"]); collection[f"relation_{name}_truth"].extend(metrics["relation"][name]["truth"])
        for name in metrics["relation"]: collection[f"relation_{name}_positive_total"].append(metrics["relation"][name]["positive_total"])
        for name in metrics["association"]: collection[f"association_{name}"].extend(metrics["association"][name])
        geometry_baseline_sum += metrics["geometry_baseline_sum"]; geometry_baseline_count += metrics["geometry_baseline_count"]
        world_rows.append({"parent_id":parent,"observations":len(world["event_index"]),"true_tokens":int(world["branch_presence_mask"].sum()),"association_pairs":len(pairs.get(parent,[]))})
    arrays={name:np.concatenate(parts) for name,parts in collection.items() if isinstance(parts,list) and parts and isinstance(parts[0],np.ndarray)}
    valid=arrays["geometry_valid"]; absolute=np.abs(arrays["geometry"]-arrays["geometry_truth"]); geometry_mae={name:float(absolute[...,i][valid[...,i]].mean()) for i,name in enumerate(("width_m","height_m","slope_deg","curvature_per_m"))}; baseline_mae={name:float(geometry_baseline_sum[i]/geometry_baseline_count[i]) for i,name in enumerate(("width_m","height_m","slope_deg","curvature_per_m"))}; axis_error=np.degrees(np.arccos(np.clip((arrays["axis"]*arrays["axis_truth"]).sum(1),-1,1)))
    return {"arrays":arrays,"event":_event_metrics(arrays["event_logits"],arrays["event_truth"]),"geometry_mae":geometry_mae,"baseline_geometry_mae":baseline_mae,"axis_mean_error_deg":float(axis_error.mean()),"token_score":np.asarray(collection["token_score"]),"token_correct":np.asarray(collection["token_correct"],dtype=bool),"token_positive_total":sum(collection["token_positive_total"]),"exact_rate":float(np.mean(collection["exact"])),"token_width_mae":float(np.mean(collection["width_error"])) if collection["width_error"] else math.inf,"token_profile_mae":float(np.mean(collection["profile_error"])) if collection["profile_error"] else math.inf,"relation":{name:{"score":np.asarray(collection[f"relation_{name}_score"]),"truth":np.asarray(collection[f"relation_{name}_truth"],dtype=bool),"positive_total":sum(collection[f"relation_{name}_positive_total"])} for name in ("persistent","reveal","withdraw")},"association":{name:np.asarray(collection[f"association_{name}"],dtype=(bool if name=="truth" else np.float64)) for name in ("place","branch","combined","truth")},"world_rows":world_rows}


def _token_metrics(split, threshold):
    metrics=_objective_recall_metrics(split["token_score"],split["token_correct"],threshold,split["token_positive_total"]); metrics["exact_rate"] = split["exact_rate"]; return metrics


def _relation_metrics(relation, threshold):
    return _objective_recall_metrics(
        relation["score"], relation["truth"], threshold, relation["positive_total"]
    )


def _structural(split):
    probability=_softmax(split["arrays"]["event_logits"],1); predicted=probability.argmax(1); truth=split["arrays"]["event_truth"]; structural=predicted!=0; score=probability[np.arange(len(predicted)),predicted]*(1-split["arrays"]["uncertainty"]); return score[structural],(predicted==truth)[structural],int((truth!=0).sum())


def _structural_metrics(values, threshold):
    score,correct,total=values; return _objective_recall_metrics(score,correct,threshold,total)


def _plot(output, summary):
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True); names=("c07","c08"); x=np.arange(2)
    axes[0,0].bar(x,[summary[n]["event"]["macro_f1"] for n in names]); axes[0,0].set(xticks=x,xticklabels=("C07","C08"),ylim=(0,1),title="A  Structure event macro-F1")
    axes[0,1].bar(x-.18,[summary[n]["tokens"]["precision"] for n in names],.36,label="precision"); axes[0,1].bar(x+.18,[summary[n]["tokens"]["recall"] for n in names],.36,label="recall"); axes[0,1].set(xticks=x,xticklabels=("C07","C08"),ylim=(0,1),title="B  Safe localized tokens"); axes[0,1].legend()
    relation=("persistent","reveal","withdraw"); axes[0,2].bar(np.arange(3)-.18,[summary["c07"]["relation"][n]["f1"] for n in relation],.36,label="C07"); axes[0,2].bar(np.arange(3)+.18,[summary["c08"]["relation"][n]["f1"] for n in relation],.36,label="C08"); axes[0,2].set(xticks=range(3),xticklabels=relation,ylim=(0,1),title="C  Transport relation F1"); axes[0,2].legend()
    axes[1,0].bar(x,[summary[n]["axis_mean_error_deg"] for n in names]); axes[1,0].axhline(10,color="red",ls="--"); axes[1,0].set(xticks=x,xticklabels=("C07","C08"),title="D  Axis error (deg)")
    axes[1,1].bar(x-.18,[summary[n]["association"]["combined"]["precision"] for n in names],.36,label="precision"); axes[1,1].bar(x+.18,[summary[n]["association"]["combined"]["recall"] for n in names],.36,label="recall"); axes[1,1].set(xticks=x,xticklabels=("C07","C08"),ylim=(0,1),title="E  Node association"); axes[1,1].legend()
    axes[1,2].bar(x-.18,[summary[n]["structural_refusal"]["precision"] for n in names],.36,label="precision"); axes[1,2].bar(x+.18,[summary[n]["structural_refusal"]["recall"] for n in names],.36,label="recall"); axes[1,2].set(xticks=x,xticklabels=("C07","C08"),ylim=(0,1),title="F  Uncertainty refusal"); axes[1,2].legend()
    for axis in axes.flat: axis.grid(axis="y",alpha=.2); axis.set_axisbelow(True)
    fig.suptitle("GSE sparse circular relation transport: C07 selection → C08 transfer")
    for suffix in ("png","pdf","svg"): fig.savefig(output/f"gse_sparse_circular_relation_transport_selection_v2.{suffix}",dpi=220)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset-root",required=True,type=Path); parser.add_argument("--sequence-manifest",required=True,type=Path); parser.add_argument("--association-pairs",required=True,type=Path); parser.add_argument("--prediction-root",action="append",required=True,type=Path); parser.add_argument("--output-dir",required=True,type=Path); args=parser.parse_args(); started=time.monotonic(); output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=False)
    if len(args.prediction_root)!=3: raise ValueError("exactly three seeds required")
    traversals=base.manifest_traversals(args.sequence_manifest.resolve()); c07=_split(7,args.dataset_root.resolve(),traversals,[p.resolve() for p in args.prediction_root],args.association_pairs.resolve()); c08=_split(8,args.dataset_root.resolve(),traversals,[p.resolve() for p in args.prediction_root],args.association_pairs.resolve())
    token_choice=_select_threshold(c07["token_score"],c07["token_correct"],TOKEN_PRECISION_FLOOR); token_threshold=math.inf if token_choice is None else token_choice["threshold"]
    relation_threshold={}; relation_metrics={"c07":{},"c08":{}}
    for name in ("persistent","reveal","withdraw"):
        choice=_select_threshold(c07["relation"][name]["score"],c07["relation"][name]["truth"],RELATION_PRECISION_FLOOR); threshold=math.inf if choice is None else choice["threshold"]; relation_threshold[name]=None if choice is None else threshold; relation_metrics["c07"][name]=_relation_metrics(c07["relation"][name],threshold); relation_metrics["c08"][name]=_relation_metrics(c08["relation"][name],threshold)
    association_threshold={}; association_metrics={"c07":{},"c08":{}}
    for name in ("place","branch","combined"):
        choice=_select_threshold(c07["association"][name],c07["association"]["truth"],ASSOCIATION_PRECISION_FLOOR,ASSOCIATION_FALSE_CEILING); threshold=math.inf if choice is None else choice["threshold"]; association_threshold[name]=None if choice is None else threshold; association_metrics["c07"][name]=_binary_metrics(c07["association"][name],c07["association"]["truth"],threshold); association_metrics["c08"][name]=_binary_metrics(c08["association"][name],c08["association"]["truth"],threshold)
    structural07=_structural(c07); structural08=_structural(c08); structural_choice=_select_threshold(structural07[0],structural07[1],STRUCTURAL_PRECISION_FLOOR,STRUCTURAL_FALSE_CEILING); structural_threshold=math.inf if structural_choice is None else structural_choice["threshold"]
    summary_split={}
    for name,split in (("c07",c07),("c08",c08)):
        summary_split[name]={"observations":len(split["arrays"]["event_truth"]),"event":split["event"],"tokens":_token_metrics(split,token_threshold),"relation":relation_metrics[name],"axis_mean_error_deg":split["axis_mean_error_deg"],"geometry_mae":split["geometry_mae"],"nonlearning_geometry_mae":split["baseline_geometry_mae"],"geometry_relative_improvement":relative_geometry_improvement(split["geometry_mae"],split["baseline_geometry_mae"]),"token_width_mae":split["token_width_mae"],"token_profile_mae":split["token_profile_mae"],"association":association_metrics[name],"structural_refusal":_structural_metrics(structural07 if name=="c07" else structural08,structural_threshold),"per_world":split["world_rows"]}
    checks={"exact_c07_c08_population":summary_split["c07"]["observations"]==21548 and summary_split["c08"]["observations"]==24394,"event_macro_f1_improves_axis_v1_by_five_points":all(summary_split[name]["event"]["macro_f1"]>=EVENT_BASELINE[name]+EVENT_GAIN for name in ("c07","c08")),"localized_tokens_safe_and_recalled":all(summary_split[name]["tokens"]["precision"]>=TOKEN_PRECISION_FLOOR and summary_split[name]["tokens"]["recall"]>=TOKEN_RECALL_FLOOR for name in ("c07","c08")),"three_transport_relations_safe_and_recalled":all(relation_metrics[split][name]["precision"]>=RELATION_PRECISION_FLOOR and relation_metrics[split][name]["recall"]>=RELATION_RECALL_FLOOR for split in ("c07","c08") for name in relation_metrics[split]),"axis_error_at_most_ten_degrees":all(summary_split[name]["axis_mean_error_deg"]<=10 for name in ("c07","c08")),"continuous_geometry_improves_nonlearning_by_ten_percent":all(summary_split[name]["geometry_relative_improvement"]["passed"] for name in ("c07","c08")),"combined_association_safe_and_recalled":all(summary_split[name]["association"]["combined"]["precision"]>=ASSOCIATION_PRECISION_FLOOR and summary_split[name]["association"]["combined"]["false_merge_fraction"]<=ASSOCIATION_FALSE_CEILING and summary_split[name]["association"]["combined"]["recall"]>=ASSOCIATION_RECALL_FLOOR for name in ("c07","c08")),"uncertainty_refusal_safe_and_recalled":all(summary_split[name]["structural_refusal"]["precision"]>=STRUCTURAL_PRECISION_FLOOR and summary_split[name]["structural_refusal"]["false_merge_fraction"]<=STRUCTURAL_FALSE_CEILING and summary_split[name]["structural_refusal"]["recall"]>=STRUCTURAL_RECALL_FLOOR for name in ("c07","c08"))}; checks={name:bool(value) for name,value in checks.items()}; scientific_pass=all(checks.values())
    summary={"schema_version":"gse_sparse_circular_relation_transport_selection_v2","status":PASS if scientific_pass else FAIL,"scientific_pass":scientific_pass,"decision":"ALLOW_RELATION_AWARE_OFFLINE_GRAPH_READINESS" if scientific_pass else "STOP_SPARSE_RELATION_TRANSPORT_BEFORE_GRAPH_AND_ATTRIBUTE","selected_on":"C07 only","transferred_once_to":"C08 zero adaptation","thresholds":{"token":None if token_choice is None else token_threshold,"relation":relation_threshold,"association":association_threshold,"structural_refusal":None if structural_choice is None else structural_threshold},**summary_split,"checks":checks,"duration_seconds":time.monotonic()-started,"optimizer_steps":0,"c08_checkpoint_observations":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0,"planner_calls":0}
    write_json(output/"summary.json",summary); write_json(output/"figure_source.json",summary)
    with (output/"per_world_metrics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=("split","parent_id","observations","true_tokens","association_pairs")); writer.writeheader()
        for split in ("c07","c08"):
            for row in summary_split[split]["per_world"]: writer.writerow({"split":split.upper(),**row})
    _plot(output,summary); print(json.dumps({"status":summary["status"],"decision":summary["decision"],"checks":checks},indent=2,sort_keys=True)); return 0 if scientific_pass else 2

if __name__=="__main__": raise SystemExit(main())
