#!/usr/bin/env python3
"""Frozen full-C07 gate for three no-slot endpoint relation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mtare_topo.data.primitive_relation_observable_batches import ObservablePrimitiveRelationBatchLoader
from mtare_topo.evaluation.primitive_relation_sparse_port_failure_attribution import exact_ranked_selection, score_quantiles
from mtare_topo.representation.primitive_endpoint_relation_metric_model import FrozenObservableEndpointRelationMetricNet, endpoint_relation_pair_score
from mtare_topo.representation.primitive_local_composition_slot_training import align_local_composition_slot_targets
from mtare_topo.representation.primitive_relation_losses import match_primitives
from mtare_topo.representation.primitive_relation_observable_model import ObservableSparsePortRelationNet
from mtare_topo.representation.primitive_relation_observable_training import observable_numpy_batch_to_torch


PASS = "PASS_PRIMITIVE_ENDPOINT_RELATION_METRIC_THREE_SEED_C07_V1"
FAIL = "FAIL_PRIMITIVE_ENDPOINT_RELATION_METRIC_THREE_SEED_C07_V1"
EXPECTED_ROWS = 64_644
EXPECTED_TASKS = 30
EXPECTED_POSITIVES = 442_936
BATCH_SIZE = 128
MINIMUM_PRECISION = .98
MINIMUM_F1_GAIN = .05


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_model(path: Path, seed: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False); contract = checkpoint.get("training_contract", {})
    if checkpoint.get("schema_version") != "primitive_endpoint_relation_metric_checkpoint_v1" or checkpoint.get("seed") != seed or contract.get("arbitrary_slots") is not False or contract.get("frozen_observable_backbone") is not True:
        raise RuntimeError("endpoint relation metric checkpoint identity drift")
    model = FrozenObservableEndpointRelationMetricNet(ObservableSparsePortRelationNet()); model.load_state_dict(checkpoint["model_state_dict"], strict=True); model.to(device).eval()
    return model, checkpoint


def _upper(device: torch.device) -> torch.Tensor:
    endpoint = torch.arange(64, device=device)
    return (endpoint[:, None] // 2 != endpoint[None, :] // 2) & torch.triu(torch.ones(64, 64, dtype=torch.bool, device=device), 1)


class RelationBuffer:
    def __init__(self) -> None:
        self.score = []; self.target = []; self.overlap = []; self.row = []; self.left = []; self.right = []

    def append(self, score, target, overlap, eligible, row_offset: int) -> None:
        indices = torch.nonzero(eligible, as_tuple=False)
        self.score.append(score[eligible].detach().cpu().numpy().astype(np.float32, copy=False))
        self.target.append(target[eligible].detach().cpu().numpy().astype(np.bool_, copy=False))
        self.overlap.append(overlap[eligible].detach().cpu().numpy().astype(np.bool_, copy=False))
        index = indices.detach().cpu().numpy()
        self.row.append((index[:, 0] + row_offset).astype(np.int32, copy=False))
        self.left.append(index[:, 1].astype(np.uint8, copy=False)); self.right.append(index[:, 2].astype(np.uint8, copy=False))

    def finalize(self, total_positive: int) -> tuple[dict, dict[str, np.ndarray]]:
        arrays = {name: np.concatenate(getattr(self, name)) for name in ("score", "target", "overlap", "row", "left", "right")}
        best = exact_ranked_selection(arrays["score"], arrays["target"], total_positive=total_positive)
        safe = exact_ranked_selection(arrays["score"], arrays["target"], total_positive=total_positive, minimum_precision=MINIMUM_PRECISION)
        metrics = {
            "candidate_pairs": int(len(arrays["score"])), "candidate_true_pairs": int(arrays["target"].sum()), "objective_true_pairs": total_positive,
            "best_f1": best, "safe": safe, "score_quantiles": score_quantiles(arrays["score"], arrays["target"]),
            "overlap_fp_at_best": int(np.count_nonzero(arrays["overlap"] & ~arrays["target"] & (arrays["score"] >= best["threshold"]))) if best["available"] else 0,
            "overlap_fp_at_safe": int(np.count_nonzero(arrays["overlap"] & ~arrays["target"] & (arrays["score"] >= safe["threshold"]))) if safe["available"] else 0,
        }
        return metrics, arrays


def complete_link_metrics(arrays: dict[str, np.ndarray], threshold: float | None, total_positive: int) -> dict:
    if threshold is None:
        return {"threshold": None, "predicted_clusters": 0, "true_positive": 0, "false_positive": 0, "false_negative": total_positive, "precision": 0.0, "recall": 0.0, "f1": 0.0, "maximum_cluster_size": 0, "clusters_larger_than4": 0, "overlap_false_positive": 0}
    selected = np.flatnonzero(arrays["score"] >= threshold)
    by_row = {}
    for index in selected:
        by_row.setdefault(int(arrays["row"][index]), []).append(index)
    tp = fp = overlap_fp = clusters_count = oversized = 0; maximum = 0
    for edges in by_row.values():
        edges.sort(key=lambda index: (-float(arrays["score"][index]), int(arrays["left"][index]), int(arrays["right"][index])))
        high = {(int(arrays["left"][index]), int(arrays["right"][index])): index for index in edges}
        endpoints = sorted({item for pair in high for item in pair}); clusters = [{item} for item in endpoints]
        for index in edges:
            left, right = int(arrays["left"][index]), int(arrays["right"][index])
            first = next(value for value in clusters if left in value); second = next(value for value in clusters if right in value)
            if first is second: continue
            if all((min(a, b), max(a, b)) in high for a in first for b in second): first.update(second); clusters.remove(second)
        for cluster in clusters:
            if len(cluster) < 2: continue
            clusters_count += 1; maximum = max(maximum, len(cluster)); oversized += int(len(cluster) > 4)
            members = sorted(cluster)
            for offset, left in enumerate(members):
                for right in members[offset + 1:]:
                    index = high[(left, right)]
                    if arrays["target"][index]: tp += 1
                    else:
                        fp += 1; overlap_fp += int(arrays["overlap"][index])
    fn = total_positive - tp; precision = tp / max(tp + fp, 1); recall = tp / max(total_positive, 1); f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"threshold": threshold, "predicted_clusters": clusters_count, "true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": precision, "recall": recall, "f1": f1, "maximum_cluster_size": maximum, "clusters_larger_than4": oversized, "overlap_false_positive": overlap_fp}


@torch.no_grad()
def evaluate_seed(model, loader, existence_threshold: float, device: torch.device) -> dict:
    upper = _upper(device); global_buffer = RelationBuffer(); task_rows = []; rows = positives = row_offset = 0
    maximum_asymmetry = 0.0; exact_symmetric = True; all_finite = True
    for task_name in loader.task_names:
        task = RelationBuffer(); task_positive = task_count = 0; length = loader._lengths[task_name]; local_offset = 0
        for start in range(0, length, BATCH_SIZE):
            indices = np.arange(start, min(start + BATCH_SIZE, length), dtype=np.int64)
            batch = observable_numpy_batch_to_torch(loader._read(task_name, indices), device=device)
            prediction = model(batch.range_valid, batch.relative_translation_current_sensor_m, batch.relative_yaw_current_sensor_deg)
            targets = align_local_composition_slot_targets(batch.targets, match_primitives(prediction.primitive, batch.targets))
            observed = targets.endpoint_supervised; validity = observed[:, :, None] & observed[:, None, :] & upper[None]
            target = (targets.labels[:, :, None] >= 0) & (targets.labels[:, :, None] == targets.labels[:, None, :]) & validity
            active = torch.sigmoid(prediction.primitive.existence_logits) >= existence_threshold; active = active.repeat_interleave(2, 1)
            eligible = validity & active[:, :, None] & active[:, None, :]; overlap = targets.disconnected_overlap & ~target
            score = endpoint_relation_pair_score(prediction)
            maximum_asymmetry = max(maximum_asymmetry, float(torch.max(torch.abs(score - score.transpose(1, 2))).cpu())); exact_symmetric &= torch.equal(score, score.transpose(1, 2)); all_finite &= bool(torch.isfinite(score).all())
            global_buffer.append(score, target, overlap, eligible, row_offset); task.append(score, target, overlap, eligible, local_offset)
            count = len(indices); value_positive = int(target.sum()); rows += count; positives += value_positive; row_offset += count; task_count += count; task_positive += value_positive; local_offset += count
        task_metrics, _ = task.finalize(task_positive); task_rows.append({"task": task_name, "rows": task_count, "objective_true_pairs": task_positive, "pair_metrics": task_metrics})
    if rows != EXPECTED_ROWS or positives != EXPECTED_POSITIVES: raise RuntimeError(f"endpoint relation metric C07 population drift: {rows}/{positives}")
    pair, arrays = global_buffer.finalize(positives); cluster = complete_link_metrics(arrays, pair["safe"]["threshold"] if pair["safe"]["available"] else None, positives)
    return {"rows": rows, "existence_threshold": existence_threshold, "objective_true_pairs": positives, "pair_metrics": pair, "complete_link_at_safe": cluster, "score_contract": {"maximum_asymmetry": maximum_asymmetry, "exact_symmetric": exact_symmetric, "all_finite": all_finite, "single_relation_threshold": True}, "tasks": task_rows}


def _plot(summary: dict, output: Path) -> None:
    seeds = summary["c07"]["seeds"]; labels = ["baseline", *[f"seed{x['seed']}" for x in seeds]]; colors = ["#777", "#287271", "#d37524", "#5b5f97"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2)); axes[0].bar(labels, [summary["c07"]["baseline_attachment_f1"], *[x["metrics"]["pair_metrics"]["best_f1"]["f1"] for x in seeds]], color=colors); axes[0].set_ylabel("C07 attachment F1")
    axes[1].bar(labels[1:], [x["metrics"]["pair_metrics"]["safe"]["true_positive"] for x in seeds], color=colors[1:]); axes[1].set_ylabel("safe true pairs")
    axes[2].bar(labels[1:], [x["metrics"]["complete_link_at_safe"]["predicted_clusters"] for x in seeds], color=colors[1:]); axes[2].set_ylabel("complete-link clusters")
    fig.suptitle("No-slot endpoint relation metric — unseen C07"); fig.tight_layout()
    for suffix in ("png", "pdf", "svg"): fig.savefig(output.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--models-root", type=Path, required=True); parser.add_argument("--sensor-root", type=Path, required=True); parser.add_argument("--teacher-root", type=Path, required=True); parser.add_argument("--observability-root", type=Path, required=True); parser.add_argument("--baseline-c07-summary", type=Path, required=True); parser.add_argument("--existence-threshold-source", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False); started = time.monotonic(); device = torch.device("cuda")
    loader = ObservablePrimitiveRelationBatchLoader(args.sensor_root / "c07", args.teacher_root / "c07", args.observability_root / "c07")
    if len(loader) != EXPECTED_ROWS or len(loader.task_names) != EXPECTED_TASKS: raise RuntimeError("endpoint relation metric C07 loader drift")
    baseline = json.loads(args.baseline_c07_summary.read_text(encoding="utf-8")); baseline_f1 = float(baseline["attachment"]["f1"])
    if baseline.get("observable_attachment_target_positives") != EXPECTED_POSITIVES: raise RuntimeError("baseline population drift")
    threshold_source = json.loads(args.existence_threshold_source.read_text(encoding="utf-8")); thresholds = {int(row["seed"]): float(row["existence_threshold"]) for row in threshold_source["seeds"]}
    seeds = []
    for seed in range(3):
        model, checkpoint = _load_model(args.models_root / f"seed{seed}/selected.pt", seed, device); metrics = evaluate_seed(model, loader, thresholds[seed], device); pair = metrics["pair_metrics"]; cluster = metrics["complete_link_at_safe"]
        checks = {"attachment_f1_gain_ge_0p05": pair["best_f1"]["f1"] - baseline_f1 >= MINIMUM_F1_GAIN, "safe_precision_ge_0p98_nonzero": pair["safe"]["available"] and pair["safe"]["precision"] >= MINIMUM_PRECISION and pair["safe"]["true_positive"] > 0, "complete_link_safe_nonzero": cluster["precision"] >= MINIMUM_PRECISION and cluster["true_positive"] > 0, "complete_link_no_overlap_or_oversize": cluster["overlap_false_positive"] == 0 and cluster["clusters_larger_than4"] == 0, "score_finite_exact_symmetric": metrics["score_contract"]["all_finite"] and metrics["score_contract"]["exact_symmetric"], "single_threshold": metrics["score_contract"]["single_relation_threshold"], "frozen_backbone": checkpoint["training_contract"]["frozen_observable_backbone"] is True}
        record = {"seed": seed, "selected_epoch": int(checkpoint["epoch"]), "metrics": metrics, "checks": checks, "pass": all(checks.values())}; seeds.append(record); _write(output / f"c07_seed{seed}.json", record); _write(output / f"c07_seed{seed}_tasks.json", metrics["tasks"]); print(json.dumps({"seed": seed, "checks": checks, "best": pair["best_f1"], "safe": pair["safe"], "cluster": cluster}), flush=True)
    passing = sum(row["pass"] for row in seeds); scientific_pass = passing >= 2
    summary = {"schema_version": "primitive_endpoint_relation_metric_three_seed_evaluation_v1", "overall_status": PASS if scientific_pass else FAIL, "scientific_pass": scientific_pass, "c07": {"baseline_attachment_f1": baseline_f1, "seeds": seeds, "passing_seeds": passing, "scientific_pass": scientific_pass}, "c08_rows_read": 0, "graph_replays": 0, "mtare_worlds_read": 0, "decision": "ALLOW_C08_ENDPOINT_RELATION_METRIC_ZERO_ADAPTATION" if scientific_pass else "STOP_ENDPOINT_RELATION_METRIC_BEFORE_C08_AND_GRAPH", "duration_seconds": time.monotonic() - started}
    _write(output / "summary.json", summary); _plot(summary, output / "primitive_endpoint_relation_metric_c07_comparison"); return 0 if scientific_pass else 2


if __name__ == "__main__": raise SystemExit(main())
