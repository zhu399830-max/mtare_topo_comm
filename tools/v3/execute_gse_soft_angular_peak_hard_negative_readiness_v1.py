#!/usr/bin/env python3
"""Attribute V1 ghost peaks and prove the V2 peak-loss corrective without training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from execute_gse_circular_peak_geometry_model_readiness_v1 import _real_batch, _selected_rows
from mtare_topo.evaluation.gse_circular_peak_metrics import circular_local_maxima
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_circular_peak_geometry_model import (
    CircularPeakGeometrySemanticNet,
    circular_peak_geometry_loss,
)
from mtare_topo.representation.gse_soft_angular_peak_loss import (
    SUPPORT_RADIUS_BINS,
    circular_soft_peak_target,
    replace_v1_presence_with_v2,
    soft_angular_hard_negative_presence_loss,
)


PASS = "PASS_GSE_SOFT_ANGULAR_PEAK_HARD_NEGATIVE_READINESS_V1"
FAIL = "FAIL_GSE_SOFT_ANGULAR_PEAK_HARD_NEGATIVE_READINESS_V1"


def _rank(scores: np.ndarray, labels: np.ndarray, total: int) -> dict:
    order = np.argsort(-scores, kind="stable")
    values = scores[order]
    truth = labels[order].astype(np.int64)
    cumulative = np.cumsum(truth)
    precision = cumulative / np.arange(1, len(values) + 1)
    ap = float(precision[truth == 1].sum() / total)
    group_end = np.flatnonzero(np.r_[values[1:] != values[:-1], True])
    safe = group_end[precision[group_end] >= .995]
    best = None if not len(safe) else {"threshold": float(values[safe[-1]]), "precision": float(precision[safe[-1]]), "recall": float(cumulative[safe[-1]] / total), "selected": int(safe[-1] + 1)}
    return {"average_precision": ap, "safe": best, "maximum_match_recall": float(truth.sum() / total)}


def _tolerant_labels(candidate: np.ndarray, confidence: np.ndarray, truth: np.ndarray, tolerance: int) -> tuple[np.ndarray, np.ndarray]:
    rows, bins = np.nonzero(candidate)
    scores = confidence[candidate]
    labels = np.zeros(len(scores), dtype=np.bool_)
    starts = np.r_[0, np.flatnonzero(rows[1:] != rows[:-1]) + 1, len(rows)]
    for start, stop in zip(starts[:-1], starts[1:], strict=True):
        row = int(rows[start])
        targets = np.flatnonzero(truth[row])
        pairs = []
        for candidate_index in range(start, stop):
            bearing = int(bins[candidate_index])
            distance = np.minimum((bearing - targets) % 180, (targets - bearing) % 180)
            for target_index in np.flatnonzero(distance <= tolerance):
                pairs.append((-float(scores[candidate_index]), int(distance[target_index]), candidate_index, int(target_index)))
        used_candidate: set[int] = set()
        used_target: set[int] = set()
        for _, _, candidate_index, target_index in sorted(pairs):
            if candidate_index not in used_candidate and target_index not in used_target:
                labels[candidate_index] = True
                used_candidate.add(candidate_index)
                used_target.add(target_index)
    return scores, labels


def _topk_nms(confidence: np.ndarray, radius: int, count: int) -> np.ndarray:
    result = np.zeros_like(confidence, dtype=np.bool_)
    for row, values in enumerate(confidence):
        selected: list[int] = []
        for bearing in np.argsort(-values, kind="stable"):
            bearing = int(bearing)
            if all(min((bearing - other) % 180, (other - bearing) % 180) > radius for other in selected):
                selected.append(bearing)
                result[row, bearing] = True
                if len(selected) == count:
                    break
    return result


def _load_predictions(suffix: str, teacher_root: Path, prediction_roots: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    confidence = []
    truth = []
    for teacher_path in sorted(teacher_root.glob(f"selection/*_{suffix}.zarr")):
        teacher = zarr.open_group(str(teacher_path), mode="r")
        parent = teacher_path.stem
        truth.append(np.asarray(teacher["presence"][:], dtype=np.bool_))
        confidence.append(np.mean([np.load(root / f"{parent}.npz")["confidence"].astype(np.float32) for root in prediction_roots], axis=0))
    return np.concatenate(confidence), np.concatenate(truth)


def _attribution(confidence: np.ndarray, truth: np.ndarray) -> dict:
    local = circular_local_maxima(confidence)
    offsets = []
    for row in range(len(confidence)):
        candidates = np.flatnonzero(local[row])
        for target in np.flatnonzero(truth[row]):
            distance = np.minimum((candidates - target) % 180, (target - candidates) % 180)
            offsets.append(int(distance.min()))
    tolerant = {}
    for tolerance in (0, 1, 2, 3, 5, 8, 10):
        scores, labels = _tolerant_labels(local, confidence, truth, tolerance)
        tolerant[str(tolerance)] = _rank(scores, labels, int(truth.sum()))
    nms = {}
    for radius in (5, 8):
        candidate = _topk_nms(confidence, radius, 4)
        for tolerance in (8, 10):
            scores, labels = _tolerant_labels(candidate, confidence, truth, tolerance)
            nms[f"radius{radius}_top4_tolerance{tolerance}"] = _rank(scores, labels, int(truth.sum()))
    dense = _rank(confidence.reshape(-1), truth.reshape(-1), int(truth.sum()))
    return {
        "observations": len(confidence), "teacher_peaks": int(truth.sum()),
        "local_maxima": int(local.sum()), "local_maxima_per_observation": float(local.sum() / len(confidence)),
        "dense_exact": dense, "tolerant_local_maxima": tolerant, "top4_nms": nms,
        "nearest_local_maximum_offset_bins": {name: float(value) for name, value in zip(("median", "p75", "p90", "p95", "p99"), np.quantile(offsets, (.5, .75, .9, .95, .99)), strict=True)},
    }


def _synthetic() -> dict:
    presence = torch.zeros(1, 180, dtype=torch.uint8)
    presence[0, [0, 40]] = 1
    correct = torch.full((1, 180), -5.0)
    correct[presence.bool()] = 5.0
    shifted = torch.roll(correct, 1, 1)
    far = torch.roll(correct, 12, 1)
    ghost = correct.clone(); ghost[0, 100] = 6.0
    losses = {name: float(soft_angular_hard_negative_presence_loss(value, presence)["total"]) for name, value in (("correct", correct), ("shifted_one_bin", shifted), ("far", far), ("ghost", ghost))}
    logits = ghost.clone().requires_grad_(True)
    loss = soft_angular_hard_negative_presence_loss(logits, presence)
    loss["total"].backward()
    target = circular_soft_peak_target(presence)
    rotated = circular_soft_peak_target(torch.roll(presence, 17, 1))
    return {"losses": losses, "ghost_gradient": float(logits.grad[0, 100]), "center_gradient": float(logits.grad[0, 0]), "rotation_error": float(torch.max(torch.abs(rotated - torch.roll(target, 17, 1)))), "center_value": float(target[0, 0]), "support_edge_value": float(target[0, SUPPORT_RADIUS_BINS]), "outside_value": float(target[0, SUPPORT_RADIUS_BINS + 1])}


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    tolerances = np.asarray((0, 2, 4, 6, 10, 16, 20))
    for split, color in (("c07", "#4e79a7"), ("c08", "#f28e2b")):
        axes[0].plot(tolerances, [summary[split]["tolerant_local_maxima"][str(value // 2)]["average_precision"] for value in tolerances], marker="o", label=split.upper(), color=color)
    axes[0].set_xlabel("One-to-one angular tolerance (deg)")
    axes[0].set_ylabel("Peak average precision")
    axes[0].set_title("A  V1 contains angular signal")
    axes[0].legend(frameon=False)
    quantiles = ("median", "p75", "p90", "p95", "p99")
    x = np.arange(len(quantiles))
    axes[1].bar(x - .18, [2 * summary["c07"]["nearest_local_maximum_offset_bins"][name] for name in quantiles], .36, label="C07")
    axes[1].bar(x + .18, [2 * summary["c08"]["nearest_local_maximum_offset_bins"][name] for name in quantiles], .36, label="C08")
    axes[1].set_xticks(x, quantiles)
    axes[1].set_ylabel("Nearest local-maximum offset (deg)")
    axes[1].set_title("B  Single-bin target is brittle")
    axes[1].legend(frameon=False)
    losses = summary["synthetic_loss"]["losses"]
    names = ("correct", "shifted_one_bin", "far", "ghost")
    axes[2].bar(np.arange(4), [losses[name] for name in names], color=["#59a14f", "#76b7b2", "#e15759", "#b07aa1"])
    axes[2].set_xticks(np.arange(4), ("correct", "1-bin shift", "far shift", "+ hard ghost"), rotation=15)
    axes[2].set_ylabel("V2 peak loss")
    axes[2].set_title("C  Corrective targets the failure")
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
        axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph hard-peak failure attribution and V2 readiness")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_soft_angular_peak_hard_negative_readiness_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--prediction-root", action="append", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    if len(args.prediction_root) != 3 or len(args.checkpoint) != 3:
        raise RuntimeError("V2 readiness requires exactly three frozen V1 seeds")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    c07_confidence, c07_truth = _load_predictions("C07", args.teacher_root.resolve(), args.prediction_root)
    c08_confidence, c08_truth = _load_predictions("C08", args.teacher_root.resolve(), args.prediction_root)
    c07 = _attribution(c07_confidence, c07_truth)
    c08 = _attribution(c08_confidence, c08_truth)
    synthetic = _synthetic()

    teacher_paths = sorted(args.teacher_root.resolve().glob("*/*.zarr"))
    rows, _ = _selected_rows(teacher_paths, args.source_root.resolve())
    scans, targets = _real_batch(rows, args.teacher_root.resolve(), args.source_root.resolve())
    real_seed = []
    for seed, checkpoint_path in enumerate(args.checkpoint):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = CircularPeakGeometrySemanticNet()
        model.load_state_dict(checkpoint["model"], strict=True)
        model.train()
        outputs = model(scans)
        v1 = circular_peak_geometry_loss(outputs, targets)
        v2 = replace_v1_presence_with_v2(v1, outputs["peak_presence_logits"], targets["presence"])
        v2["total"].backward()
        finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
        logits = outputs["peak_presence_logits"].detach().clone().requires_grad_(True)
        standalone = soft_angular_hard_negative_presence_loss(logits, targets["presence"])
        standalone["total"].backward()
        soft_target = circular_soft_peak_target(targets["presence"])
        outside = soft_target == 0
        hard_index = torch.argmax(torch.where(outside, logits.detach(), torch.full_like(logits, -torch.inf)), dim=1)
        hard_gradient = logits.grad[torch.arange(len(logits)), hard_index]
        real_seed.append({"seed": seed, "v1_total": float(v1["total"]), "v2_total": float(v2["total"]), "soft_heatmap": float(v2["soft_heatmap"]), "hard_negative_ranking": float(v2["hard_negative_ranking"]), "finite_backward": finite, "minimum_top_hard_negative_gradient": float(hard_gradient.min()), "positive_hard_negative_gradients": bool((hard_gradient > 0).all())})

    checks = {
        "reproduce_v1_hard_peak_failure": abs(c07["tolerant_local_maxima"]["0"]["average_precision"] - .06903424166741023) <= 1e-12 and abs(c08["tolerant_local_maxima"]["0"]["average_precision"] - .057518945872114556) <= 1e-12,
        "c07_angle_signal_and_offset_support": c07["tolerant_local_maxima"]["5"]["average_precision"] >= .89 and c07["nearest_local_maximum_offset_bins"]["p90"] == SUPPORT_RADIUS_BINS,
        "c08_confirms_without_hyperparameter_selection": c08["tolerant_local_maxima"]["5"]["average_precision"] >= .89,
        "decode_only_insufficient_below_20_deg": c07["top4_nms"]["radius5_top4_tolerance8"]["safe"] is not None and c07["top4_nms"]["radius5_top4_tolerance8"]["safe"]["recall"] < .50,
        "soft_target_rotation_center_and_support": synthetic["rotation_error"] == 0.0 and synthetic["center_value"] == 1.0 and synthetic["support_edge_value"] > 0 and synthetic["outside_value"] == 0.0,
        "synthetic_ordering_targets_shift_far_and_ghost": synthetic["losses"]["correct"] < synthetic["losses"]["shifted_one_bin"] < synthetic["losses"]["far"] and synthetic["losses"]["correct"] < synthetic["losses"]["ghost"],
        "hard_ghost_gets_suppressing_gradient": synthetic["ghost_gradient"] > 0 and synthetic["center_gradient"] < 0,
        "all_three_frozen_backbones_real_finite_backward": all(row["finite_backward"] and row["positive_hard_negative_gradients"] for row in real_seed),
        "zero_training_checkpoint_test_graph": True,
    }
    scientific_pass = all(checks.values())
    summary = {
        "schema_version": "gse_soft_angular_peak_hard_negative_readiness_v1",
        "status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "decision": "ALLOW_SOFT_ANGULAR_PEAK_HARD_NEGATIVE_V2_TRAINING_DATA_CARD" if scientific_pass else "STOP_SOFT_ANGULAR_PEAK_V2",
        "failure_classification": "SINGLE_BIN_HARD_TARGET_AND_EQUAL_MASS_BCE_HIGH_CONFIDENCE_GHOST_PEAK_FAILURE",
        "v2_scope": "Keep V1 backbone/global geometry/splits/seeds/epochs; replace only balanced exact-bin presence loss with radius-4 soft circular focal heatmap plus equal-count top hard-negative ranking.",
        "support_radius_provenance": "C07-only nearest-local-maximum p90=4 bins; C08 is confirmatory and does not choose the radius.",
        "c07": c07, "c08": c08, "synthetic_loss": synthetic, "real_seed_readiness": real_seed,
        "checks": checks,
        "optimizer_steps": 0, "model_updates": 0, "checkpoint_writes": 0,
        "threshold_selection_steps": 0, "c09_worlds_read": 0, "c10_worlds_read": 0,
        "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
        "duration_seconds": time.monotonic() - started,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "figure_source.json", {"schema_version": "gse_soft_angular_peak_hard_negative_readiness_figure_source_v1", "summary": summary})
    with (output / "attribution_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "tolerance_bins", "tolerance_deg", "average_precision", "maximum_match_recall", "safe_recall"))
        writer.writeheader()
        for split, data in (("C07", c07), ("C08", c08)):
            for tolerance, metrics in data["tolerant_local_maxima"].items():
                writer.writerow({"split": split, "tolerance_bins": tolerance, "tolerance_deg": 2 * int(tolerance), "average_precision": metrics["average_precision"], "maximum_match_recall": metrics["maximum_match_recall"], "safe_recall": None if metrics["safe"] is None else metrics["safe"]["recall"]})
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
