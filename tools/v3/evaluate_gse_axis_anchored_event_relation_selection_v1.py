#!/usr/bin/env python3
"""C07 selection and one C08 transfer for axis-anchored event relations."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import train_gse_axis_anchored_event_relation_v1 as train
from mtare_topo.evaluation.gse_metrics import relative_geometry_improvement
from mtare_topo.governance import write_json
from mtare_topo.semantics.range_geometry_baseline import RangeGeometryBaseline


PASS = "PASS_GSE_AXIS_ANCHORED_EVENT_RELATION_SELECTION_V1"
FAIL = "FAIL_GSE_AXIS_ANCHORED_EVENT_RELATION_SELECTION_V1"
EVENT_BASELINE_MACRO_F1 = 0.6879041031973032
EVENT_REQUIRED_MACRO_F1 = EVENT_BASELINE_MACRO_F1 + 0.05
RELATION_PRECISION_FLOOR = 0.98
RELATION_RECALL_FLOOR = 0.25
BRANCH_PRECISION_FLOOR = 0.995
BRANCH_RECALL_FLOOR = 0.50
ASSOCIATION_PRECISION_FLOOR = 0.98
ASSOCIATION_FALSE_MERGE_CEILING = 0.01
ASSOCIATION_RECALL_FLOOR = 0.25
STRUCTURAL_PRECISION_FLOOR = 0.98
STRUCTURAL_FALSE_ACCEPT_CEILING = 0.01
STRUCTURAL_RECALL_FLOOR = 0.25


def _normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12)


def _binary_metrics(score: np.ndarray, truth: np.ndarray, threshold: float) -> dict:
    score = np.asarray(score, dtype=np.float64); truth = np.asarray(truth, dtype=bool)
    selected = score >= float(threshold); tp = int((selected & truth).sum()); fp = int((selected & ~truth).sum()); fn = int((~selected & truth).sum()); tn = int((~selected & ~truth).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1); f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"threshold": float(threshold), "precision": float(precision), "recall": float(recall), "f1": float(f1), "false_merge_fraction": float(fp / max(tp + fp, 1)), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accepted": tp + fp, "positives": tp + fn}


def _select_threshold(score: np.ndarray, truth: np.ndarray, *, precision_floor: float, false_ceiling: float | None = None) -> dict | None:
    score = np.asarray(score, dtype=np.float64); truth = np.asarray(truth, dtype=bool)
    if score.ndim != 1 or truth.shape != score.shape or not truth.any() or not (~truth).any() or not np.all(np.isfinite(score)):
        raise ValueError("threshold population is invalid")
    positive = np.sort(score[truth]); negative = np.sort(score[~truth]); candidates = np.unique(positive)
    tp = len(positive) - np.searchsorted(positive, candidates, side="left"); fp = len(negative) - np.searchsorted(negative, candidates, side="left")
    precision = tp / np.maximum(tp + fp, 1); recall = tp / len(positive); valid = precision >= precision_floor
    if false_ceiling is not None: valid &= fp / np.maximum(tp + fp, 1) <= false_ceiling
    if not valid.any(): return None
    indices = np.flatnonzero(valid); best = indices[np.lexsort((-precision[indices], -recall[indices]))[0]]
    return _binary_metrics(score, truth, float(candidates[best]))


def _event_metrics(logits: np.ndarray, truth: np.ndarray) -> dict:
    predicted = np.asarray(logits).argmax(1); truth = np.asarray(truth, dtype=np.int64); matrix = np.zeros((5, 5), dtype=np.int64); np.add.at(matrix, (truth, predicted), 1)
    names = ("corridor", "junction", "terminal", "turn", "geometry_transition"); values = []; per_class = {}
    for index, name in enumerate(names):
        tp = int(matrix[index, index]); p = int(matrix[:, index].sum()); n = int(matrix[index].sum()); precision = tp / p if p else 0.0; recall = tp / n if n else 0.0; f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values.append(f1); per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": n}
    return {"macro_f1": float(np.mean(values)), "accuracy": float(np.trace(matrix) / matrix.sum()), "per_class": per_class, "confusion": matrix.tolist()}


def _softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64); value -= value.max(1, keepdims=True); value = np.exp(value); return value / value.sum(1, keepdims=True)


def _load_pairs(path: Path, condition: int) -> dict[str, list[dict]]:
    result = defaultdict(list)
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line); match = re.search(r"_C(\d+)$", str(record["parent_id"]))
            if match is not None and int(match.group(1)) == condition: result[str(record["parent_id"])].append(record)
    return dict(result)


def _split(
    condition: int, dataset_root: Path, traversals: dict[str, list[str]], prediction_roots: list[Path], pair_manifest: Path,
) -> dict:
    parents = sorted(parent for parent in traversals if parent.endswith(f"_C{condition:02d}")); pairs = _load_pairs(pair_manifest, condition)
    expected_rows = 21548 if condition == 7 else 24394; expected_branches = 45504 if condition == 7 else 51537
    collection = defaultdict(list); seed_collection = [defaultdict(list) for _ in range(3)]; pair_scores = {"place": [], "branch": [], "combined": [], "truth": []}
    geometry_baseline_sum = np.zeros(4); geometry_baseline_count = np.zeros(4, dtype=np.int64); baseline = RangeGeometryBaseline(); per_world = []
    for parent in parents:
        world = train._load_world(dataset_root, parent, traversals[parent]); predictions = [dict(np.load(root / f"{parent}.npz")) for root in prediction_roots]
        for predicted in predictions:
            if not np.array_equal(predicted["global_sequence_index"], world["global_sequence_index"]): raise RuntimeError(f"prediction join drift: {parent}")
        ensemble = {
            "event_logits": np.mean([p["event_logits"].astype(np.float64) for p in predictions], axis=0),
            "relation": np.mean([p["relation_probability_sequence"].astype(np.float64) for p in predictions], axis=0),
            "branch_union": np.mean([p["branch_union_probability"].astype(np.float64) for p in predictions], axis=0),
            "geometry": np.mean([p["geometry"].astype(np.float64) for p in predictions], axis=0),
            "axis": _normalize(np.mean([p["local_axis"].astype(np.float64) for p in predictions], axis=0)),
            "place": _normalize(np.mean([p["place_descriptor"].astype(np.float64) for p in predictions], axis=0)),
            "uncertainty": np.mean([p["observation_uncertainty"].astype(np.float64) for p in predictions], axis=0),
            "branch_descriptor": _normalize(np.mean([p["branch_descriptor"].astype(np.float64) for p in predictions], axis=0)),
            "branch_bins": predictions[0]["branch_bin_index"].astype(np.int16),
        }
        for p in predictions[1:]:
            if not np.array_equal(p["branch_bin_index"], ensemble["branch_bins"]): raise RuntimeError("teacher-bin prediction packing drift")
        collection["event_logits"].append(ensemble["event_logits"]); collection["event_truth"].append(world["event_index"]); collection["relation"].append(ensemble["relation"]); collection["relation_truth"].append(world["relation_index"])
        collection["branch_union"].append(ensemble["branch_union"]); collection["branch_truth"].append(world["branch_presence_mask"]); collection["geometry"].append(ensemble["geometry"]); collection["geometry_truth"].append(world["geometry"]); collection["geometry_valid"].append(world["geometry_valid_mask"]); collection["axis"].append(ensemble["axis"]); collection["axis_truth"].append(world["local_axis"]); collection["uncertainty"].append(ensemble["uncertainty"])
        for seed, predicted in enumerate(predictions):
            seed_collection[seed]["event_logits"].append(predicted["event_logits"].astype(np.float64)); seed_collection[seed]["geometry"].append(predicted["geometry"].astype(np.float64)); seed_collection[seed]["axis"].append(predicted["local_axis"].astype(np.float64))
        for row in range(len(world["event_index"])):
            current_frame = int(world["references"][row, -1]); estimate = baseline.predict(world["range_m"][current_frame], world["valid_mask"][current_frame]); predicted_geometry = np.asarray([estimate[name] for name in ("width_m", "height_m", "slope_deg", "curvature_per_m")]); valid = world["geometry_valid_mask"][row]
            geometry_baseline_sum += np.abs(predicted_geometry - world["geometry"][row]) * valid; geometry_baseline_count += valid
        row_by_global = {int(value): row for row, value in enumerate(world["global_sequence_index"])}
        parent_pair_count = 0
        for pair in pairs.get(parent, []):
            left = row_by_global[int(pair["anchor_global_sequence_index"])]; right = row_by_global[int(pair["paired_global_sequence_index"])]
            place = float(ensemble["place"][left] @ ensemble["place"][right]); left_valid = ensemble["branch_bins"][left] >= 0; right_valid = ensemble["branch_bins"][right] >= 0
            branch = float((ensemble["branch_descriptor"][left, left_valid] @ ensemble["branch_descriptor"][right, right_valid].T).max())
            pair_scores["place"].append(place); pair_scores["branch"].append(branch); pair_scores["combined"].append(0.5 * (place + branch)); pair_scores["truth"].append(bool(pair["same_identity"])); parent_pair_count += 1
        per_world.append({"parent_id": parent, "observations": len(world["event_index"]), "branches": int(world["branch_presence_mask"].sum()), "association_pairs": parent_pair_count})
    arrays = {name: np.concatenate(parts) for name, parts in collection.items()}; seed_arrays = [{name: np.concatenate(parts) for name, parts in values.items()} for values in seed_collection]
    if len(arrays["event_truth"]) != expected_rows or int(arrays["branch_truth"].sum()) != expected_branches: raise RuntimeError(f"C{condition:02d} population drift")
    relation = {}; relation_score = {}; relation_truth = {}
    for channel, name in enumerate(("persistent", "reveal", "withdraw")):
        valid = arrays["relation_truth"][..., channel] >= 0; relation_score[name] = arrays["relation"][..., channel][valid]; relation_truth[name] = arrays["relation_truth"][..., channel][valid] == 1
    geometry_mae = {}; seed_metrics = []
    for seed, values in enumerate(seed_arrays):
        valid = arrays["geometry_valid"]; absolute = np.abs(values["geometry"] - arrays["geometry_truth"]); seed_geometry = {name: float(absolute[..., index][valid[..., index]].mean()) for index, name in enumerate(("width_m", "height_m", "slope_deg", "curvature_per_m"))}
        axis_error = np.degrees(np.arccos(np.clip((_normalize(values["axis"]) * arrays["axis_truth"]).sum(1), -1, 1)))
        seed_metrics.append({"seed": seed, "event": _event_metrics(values["event_logits"], arrays["event_truth"]), "geometry_mae": seed_geometry, "axis_mean_error_deg": float(axis_error.mean())})
    valid = arrays["geometry_valid"]; absolute = np.abs(arrays["geometry"] - arrays["geometry_truth"])
    for index, name in enumerate(("width_m", "height_m", "slope_deg", "curvature_per_m")): geometry_mae[name] = float(absolute[..., index][valid[..., index]].mean())
    axis_error = np.degrees(np.arccos(np.clip((arrays["axis"] * arrays["axis_truth"]).sum(1), -1, 1)))
    return {
        "condition": condition, "arrays": arrays, "relation_score": relation_score, "relation_truth": relation_truth,
        "branch_score": arrays["branch_union"].reshape(-1), "branch_truth_flat": arrays["branch_truth"].reshape(-1),
        "association": {name: np.asarray(value, dtype=(bool if name == "truth" else np.float64)) for name, value in pair_scores.items()},
        "event": _event_metrics(arrays["event_logits"], arrays["event_truth"]), "seed_metrics": seed_metrics,
        "geometry_mae": geometry_mae, "nonlearning_geometry_mae": {name: float(geometry_baseline_sum[index] / geometry_baseline_count[index]) for index, name in enumerate(("width_m", "height_m", "slope_deg", "curvature_per_m"))},
        "axis_mean_error_deg": float(axis_error.mean()), "per_world": per_world,
    }


def _structural_scores(split: dict) -> tuple[np.ndarray, np.ndarray, int]:
    probabilities = _softmax(split["arrays"]["event_logits"]); predicted = probabilities.argmax(1); truth = split["arrays"]["event_truth"]; structural = predicted != 0
    reliability = probabilities[np.arange(len(predicted)), predicted] * (1.0 - split["arrays"]["uncertainty"]); correct = predicted == truth
    return reliability[structural], correct[structural], int((truth != 0).sum())


def _structural_metrics(score: np.ndarray, correct: np.ndarray, truth_structural_count: int, threshold: float) -> dict:
    selected = score >= threshold; tp = int((selected & correct).sum()); fp = int((selected & ~correct).sum()); precision = tp / max(tp + fp, 1); recall = tp / max(truth_structural_count, 1)
    return {"threshold": float(threshold), "precision": float(precision), "false_accept_fraction": float(fp / max(tp + fp, 1)), "recall": float(recall), "accepted": tp + fp, "correct": tp, "false": fp, "true_structural": truth_structural_count}


def _plot(output: Path, summary: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.2), constrained_layout=True); splits = ("c07", "c08"); x = np.arange(2)
    axes[0, 0].bar(x, [summary[s]["event"]["macro_f1"] for s in splits], color="#4e79a7"); axes[0, 0].axhline(EVENT_REQUIRED_MACRO_F1, color="#e15759", linestyle="--"); axes[0, 0].set(xticks=x, xticklabels=("C07", "C08"), ylim=(0, 1), title="A  Five-event macro-F1")
    names = ("persistent", "reveal", "withdraw"); width = .35
    axes[0, 1].bar(np.arange(3)-width/2, [summary["c07"]["relation"][n]["f1"] for n in names], width, label="C07"); axes[0, 1].bar(np.arange(3)+width/2, [summary["c08"]["relation"][n]["f1"] for n in names], width, label="C08"); axes[0, 1].set(xticks=range(3), xticklabels=names, ylim=(0,1), title="B  Temporal relation F1"); axes[0, 1].legend(frameon=False)
    axes[0, 2].bar(x-.18, [summary[s]["branch_presence"]["precision"] for s in splits], .36, label="precision"); axes[0, 2].bar(x+.18, [summary[s]["branch_presence"]["recall"] for s in splits], .36, label="recall"); axes[0, 2].set(xticks=x, xticklabels=("C07","C08"), ylim=(0,1), title="C  Current branch field"); axes[0, 2].legend(frameon=False)
    geometry = ("width_m", "height_m", "slope_deg", "curvature_per_m")
    axes[1, 0].bar(np.arange(4)-.18, [summary["c07"]["geometry_relative_improvement"]["per_field_relative_improvement"][n] for n in geometry], .36, label="C07"); axes[1, 0].bar(np.arange(4)+.18, [summary["c08"]["geometry_relative_improvement"]["per_field_relative_improvement"][n] for n in geometry], .36, label="C08"); axes[1, 0].axhline(.1, color="#e15759", linestyle="--"); axes[1, 0].set(xticks=range(4), xticklabels=("width","height","slope","curve"), title="D  Gain over rule geometry"); axes[1, 0].legend(frameon=False)
    association = ("place", "branch", "combined")
    axes[1, 1].bar(np.arange(3)-.18, [summary["c07"]["association"][n]["precision"] for n in association], .36, label="C07"); axes[1, 1].bar(np.arange(3)+.18, [summary["c08"]["association"][n]["precision"] for n in association], .36, label="C08"); axes[1, 1].axhline(.99, color="#e15759", linestyle="--"); axes[1, 1].set(xticks=range(3), xticklabels=association, ylim=(0,1.02), title="E  Association precision"); axes[1, 1].legend(frameon=False)
    axes[1, 2].bar(x-.18, [summary[s]["structural_refusal"]["precision"] for s in splits], .36, label="precision"); axes[1, 2].bar(x+.18, [summary[s]["structural_refusal"]["recall"] for s in splits], .36, label="recall"); axes[1, 2].set(xticks=x, xticklabels=("C07","C08"), ylim=(0,1), title="F  Uncertainty-refused events"); axes[1, 2].legend(frameon=False)
    for axis in axes.flat: axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True)
    fig.suptitle("GSE-Graph axis-anchored event–relation capacity")
    for suffix in ("png", "pdf", "svg"): fig.savefig(output / f"gse_axis_anchored_event_relation_selection_v1.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset-root", required=True, type=Path); parser.add_argument("--sequence-manifest", required=True, type=Path); parser.add_argument("--association-pairs", required=True, type=Path); parser.add_argument("--prediction-root", action="append", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); started = time.monotonic(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    if len(args.prediction_root) != 3: raise ValueError("selection requires exactly three seed prediction roots")
    traversals = train.manifest_traversals(args.sequence_manifest.resolve()); c07 = _split(7, args.dataset_root.resolve(), traversals, [p.resolve() for p in args.prediction_root], args.association_pairs.resolve()); c08 = _split(8, args.dataset_root.resolve(), traversals, [p.resolve() for p in args.prediction_root], args.association_pairs.resolve())
    relation_threshold = {}; relation_metrics = {"c07": {}, "c08": {}}
    for name in ("persistent", "reveal", "withdraw"):
        choice = _select_threshold(c07["relation_score"][name], c07["relation_truth"][name], precision_floor=RELATION_PRECISION_FLOOR)
        if choice is None: relation_threshold[name] = None; relation_metrics["c07"][name] = _binary_metrics(c07["relation_score"][name], c07["relation_truth"][name], math.inf); relation_metrics["c08"][name] = _binary_metrics(c08["relation_score"][name], c08["relation_truth"][name], math.inf)
        else:
            relation_threshold[name] = choice["threshold"]; relation_metrics["c07"][name] = choice; relation_metrics["c08"][name] = _binary_metrics(c08["relation_score"][name], c08["relation_truth"][name], choice["threshold"])
    branch_choice = _select_threshold(c07["branch_score"], c07["branch_truth_flat"], precision_floor=BRANCH_PRECISION_FLOOR); branch_threshold = None if branch_choice is None else branch_choice["threshold"]
    branch_metrics = {"c07": _binary_metrics(c07["branch_score"], c07["branch_truth_flat"], math.inf if branch_threshold is None else branch_threshold), "c08": _binary_metrics(c08["branch_score"], c08["branch_truth_flat"], math.inf if branch_threshold is None else branch_threshold)}
    association_threshold = {}; association_metrics = {"c07": {}, "c08": {}}
    for name in ("place", "branch", "combined"):
        choice = _select_threshold(c07["association"][name], c07["association"]["truth"], precision_floor=ASSOCIATION_PRECISION_FLOOR, false_ceiling=ASSOCIATION_FALSE_MERGE_CEILING); threshold = None if choice is None else choice["threshold"]; association_threshold[name] = threshold
        association_metrics["c07"][name] = _binary_metrics(c07["association"][name], c07["association"]["truth"], math.inf if threshold is None else threshold); association_metrics["c08"][name] = _binary_metrics(c08["association"][name], c08["association"]["truth"], math.inf if threshold is None else threshold)
    structural07 = _structural_scores(c07); structural08 = _structural_scores(c08); structural_choice = _select_threshold(structural07[0], structural07[1], precision_floor=STRUCTURAL_PRECISION_FLOOR, false_ceiling=STRUCTURAL_FALSE_ACCEPT_CEILING); structural_threshold = None if structural_choice is None else structural_choice["threshold"]
    structural = {"c07": _structural_metrics(*structural07, math.inf if structural_threshold is None else structural_threshold), "c08": _structural_metrics(*structural08, math.inf if structural_threshold is None else structural_threshold)}
    split_summary = {}
    for name, split in (("c07", c07), ("c08", c08)):
        improvement = relative_geometry_improvement(split["geometry_mae"], split["nonlearning_geometry_mae"])
        split_summary[name] = {"observations": len(split["arrays"]["event_truth"]), "event": split["event"], "seed_metrics": split["seed_metrics"], "relation": relation_metrics[name], "branch_presence": branch_metrics[name], "geometry_mae": split["geometry_mae"], "nonlearning_geometry_mae": split["nonlearning_geometry_mae"], "geometry_relative_improvement": improvement, "axis_mean_error_deg": split["axis_mean_error_deg"], "association": association_metrics[name], "structural_refusal": structural[name], "per_world": split["per_world"]}
    checks = {
        "exact_c07_c08_population": split_summary["c07"]["observations"] == 21548 and split_summary["c08"]["observations"] == 24394,
        "event_macro_f1_gain_at_least_five_points": all(split_summary[name]["event"]["macro_f1"] >= EVENT_REQUIRED_MACRO_F1 for name in ("c07", "c08")),
        "three_relation_channels_safe_and_nontrivial": all(relation_metrics[split][name]["precision"] >= RELATION_PRECISION_FLOOR and relation_metrics[split][name]["recall"] >= RELATION_RECALL_FLOOR for split in ("c07", "c08") for name in ("persistent", "reveal", "withdraw")),
        "current_branch_field_safe_and_recalled": all(branch_metrics[split]["precision"] >= BRANCH_PRECISION_FLOOR and branch_metrics[split]["recall"] >= BRANCH_RECALL_FLOOR for split in ("c07", "c08")),
        "continuous_geometry_improves_rule_baseline": all(split_summary[name]["geometry_relative_improvement"]["passed"] for name in ("c07", "c08")),
        "axis_error_at_most_ten_degrees": all(split_summary[name]["axis_mean_error_deg"] <= 10.0 for name in ("c07", "c08")),
        "combined_association_safe_and_recalled": all(association_metrics[split]["combined"]["precision"] >= ASSOCIATION_PRECISION_FLOOR and association_metrics[split]["combined"]["false_merge_fraction"] <= ASSOCIATION_FALSE_MERGE_CEILING and association_metrics[split]["combined"]["recall"] >= ASSOCIATION_RECALL_FLOOR for split in ("c07", "c08")),
        "uncertainty_refusal_safe_and_recalled": all(structural[split]["precision"] >= STRUCTURAL_PRECISION_FLOOR and structural[split]["false_accept_fraction"] <= STRUCTURAL_FALSE_ACCEPT_CEILING and structural[split]["recall"] >= STRUCTURAL_RECALL_FLOOR for split in ("c07", "c08")),
    }
    checks = {name: bool(value) for name, value in checks.items()}; scientific_pass = all(checks.values())
    summary = {"schema_version": "gse_axis_anchored_event_relation_selection_v1", "status": PASS if scientific_pass else FAIL, "scientific_pass": scientific_pass, "decision": "ALLOW_RELATION_AWARE_OFFLINE_GRAPH_READINESS" if scientific_pass else "STOP_AXIS_ANCHORED_EVENT_RELATION_BEFORE_GRAPH_AND_ATTRIBUTE", "selected_on": "C07 only", "transferred_once_to": "C08 zero adaptation", "baseline": {"event_macro_f1": EVENT_BASELINE_MACRO_F1, "required_event_macro_f1": EVENT_REQUIRED_MACRO_F1, "geometry": "deterministic RangeGeometryBaseline on the same current scans"}, "thresholds": {"relation": relation_threshold, "branch_presence": branch_threshold, "association": association_threshold, "structural_refusal": structural_threshold}, **split_summary, "checks": checks, "duration_seconds": time.monotonic()-started, "optimizer_steps": 0, "c08_checkpoint_observations": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0}
    write_json(output / "summary.json", summary); write_json(output / "figure_source.json", summary)
    with (output / "per_world_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "parent_id", "observations", "branches", "association_pairs")); writer.writeheader()
        for split in ("c07", "c08"):
            for row in split_summary[split]["per_world"]: writer.writerow({"split": split.upper(), **row})
    _plot(output, summary); print(json.dumps({"status": summary["status"], "decision": summary["decision"], "checks": checks}, indent=2, sort_keys=True)); return 0 if scientific_pass else 2


if __name__ == "__main__": raise SystemExit(main())
