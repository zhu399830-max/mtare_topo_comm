#!/usr/bin/env python3
"""Frozen full-C07 gate for three local composition-slot checkpoints."""

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

from mtare_topo.data.primitive_relation_observable_batches import (
    ObservablePrimitiveRelationBatchLoader,
)
from mtare_topo.evaluation.primitive_relation_sparse_port_failure_attribution import (
    exact_ranked_selection,
    score_quantiles,
)
from mtare_topo.representation.primitive_local_composition_slot_decoding import (
    structured_endpoint_slot_confidence,
    structured_same_slot_pair_score,
)
from mtare_topo.representation.primitive_local_composition_slot_model import (
    FrozenObservableLocalCompositionSlotNet,
)
from mtare_topo.representation.primitive_local_composition_slot_training import (
    align_local_composition_slot_targets,
)
from mtare_topo.representation.primitive_relation_losses import match_primitives
from mtare_topo.representation.primitive_relation_observable_model import (
    ObservableSparsePortRelationNet,
)
from mtare_topo.representation.primitive_relation_observable_training import (
    observable_numpy_batch_to_torch,
)


PASS = "PASS_PRIMITIVE_LOCAL_COMPOSITION_SLOT_THREE_SEED_C07_V1"
FAIL = "FAIL_PRIMITIVE_LOCAL_COMPOSITION_SLOT_THREE_SEED_C07_V1"
EXPECTED_ROWS = 64_644
EXPECTED_TASKS = 30
EXPECTED_OBSERVABLE_POSITIVES = 442_936
EVALUATION_BATCH_SIZE = 128
MINIMUM_SAFE_PRECISION = 0.98
MINIMUM_F1_GAIN = 0.05


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_model(path: Path, *, seed: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    contract = checkpoint.get("training_contract", {})
    if (
        checkpoint.get("schema_version") != "primitive_local_composition_slot_checkpoint_v1"
        or int(checkpoint.get("seed", -1)) != seed
        or contract.get("teacher_relation_representation")
        != "lossless_exchangeable_composition_cliques"
        or contract.get("composition_slot_count") != 32
        or contract.get("frozen_observable_backbone") is not True
    ):
        raise RuntimeError("local composition-slot checkpoint identity drift")
    model = FrozenObservableLocalCompositionSlotNet(ObservableSparsePortRelationNet())
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


class PairBuffer:
    def __init__(self) -> None:
        self.score: list[np.ndarray] = []
        self.target: list[np.ndarray] = []
        self.overlap: list[np.ndarray] = []

    def append(self, score: torch.Tensor, target: torch.Tensor, overlap: torch.Tensor) -> None:
        self.score.append(score.detach().cpu().numpy().astype(np.float32, copy=False))
        self.target.append(target.detach().cpu().numpy().astype(np.bool_, copy=False))
        self.overlap.append(overlap.detach().cpu().numpy().astype(np.bool_, copy=False))

    def finalize(self, *, total_positive: int) -> dict:
        score = np.concatenate(self.score) if self.score else np.empty(0, dtype=np.float32)
        target = np.concatenate(self.target) if self.target else np.empty(0, dtype=np.bool_)
        overlap = np.concatenate(self.overlap) if self.overlap else np.empty(0, dtype=np.bool_)
        best = exact_ranked_selection(score, target, total_positive=total_positive)
        safe = exact_ranked_selection(
            score, target, total_positive=total_positive,
            minimum_precision=MINIMUM_SAFE_PRECISION,
        )
        return {
            "best_f1": best,
            "safe": safe,
            "candidate_pairs": int(len(target)),
            "candidate_true_pairs": int(np.count_nonzero(target)),
            "objective_true_pairs": int(total_positive),
            "score_quantiles": score_quantiles(score, target),
            "overlap_hard_negative_fp_at_best_f1": int(np.count_nonzero(
                overlap & ~target & (score >= best["threshold"])
            )) if best["available"] else 0,
            "overlap_hard_negative_fp_at_safe": int(np.count_nonzero(
                overlap & ~target & (score >= safe["threshold"])
            )) if safe["available"] else 0,
        }


class SlotBuffer:
    def __init__(self) -> None:
        self.best_slot: list[np.ndarray] = []
        self.confidence: list[np.ndarray] = []
        self.active: list[np.ndarray] = []
        self.teacher_clusters: list[np.ndarray] = []

    def append(
        self,
        best_slot: torch.Tensor,
        confidence: torch.Tensor,
        active: torch.Tensor,
        teacher_clusters: torch.Tensor,
    ) -> None:
        self.best_slot.append(best_slot.detach().cpu().numpy().astype(np.uint8, copy=False))
        self.confidence.append(confidence.detach().cpu().numpy().astype(np.float32, copy=False))
        self.active.append(active.detach().cpu().numpy().astype(np.bool_, copy=False))
        self.teacher_clusters.append(teacher_clusters.detach().cpu().numpy().astype(np.int16, copy=False))

    def finalize(self, *, threshold: float | None) -> dict:
        best = np.concatenate(self.best_slot)
        confidence = np.concatenate(self.confidence)
        active = np.concatenate(self.active)
        teacher = np.concatenate(self.teacher_clusters)
        if threshold is None:
            accepted = np.zeros_like(active)
        else:
            accepted = active & (confidence >= float(threshold))
        clusters = []; maximum = []; endpoints = []
        oversized = collapsed = 0
        for row in range(len(best)):
            counts = np.bincount(best[row, accepted[row]], minlength=32)
            nontrivial = counts[counts >= 2]
            clusters.append(len(nontrivial)); maximum.append(int(nontrivial.max()) if len(nontrivial) else 0)
            endpoints.append(int(accepted[row].sum()))
            oversized += int(np.count_nonzero(nontrivial > 4))
            collapsed += int(endpoints[-1] >= 4 and maximum[-1] / endpoints[-1] > 0.75)
        return {
            "threshold": threshold,
            "rows": int(len(best)),
            "accepted_endpoints": int(np.sum(endpoints)),
            "rows_with_candidate_cluster": int(np.count_nonzero(np.asarray(clusters) > 0)),
            "predicted_nontrivial_clusters": int(np.sum(clusters)),
            "mean_predicted_nontrivial_clusters_per_row": float(np.mean(clusters)),
            "teacher_clusters": int(teacher.sum()),
            "mean_teacher_clusters_per_row": float(teacher.mean()),
            "maximum_predicted_cluster_size": int(max(maximum, default=0)),
            "predicted_clusters_larger_than_teacher_max4": oversized,
            "single_slot_collapse_rows": collapsed,
            "mean_accepted_endpoints_per_row": float(np.mean(endpoints)),
            "endpoint_confidence_quantiles": {
                key: float(np.quantile(confidence[active], quantile))
                for key, quantile in (("q50", .5), ("q90", .9), ("q99", .99))
            },
        }


def _cross_primitive_upper(device: torch.device) -> torch.Tensor:
    endpoint = torch.arange(64, device=device); primitive = endpoint // 2
    return (
        (primitive[:, None] != primitive[None, :])
        & torch.triu(torch.ones(64, 64, dtype=torch.bool, device=device), diagonal=1)
    )


@torch.no_grad()
def evaluate_seed(
    model,
    loader,
    *,
    existence_threshold: float,
    device: torch.device,
) -> dict:
    global_pairs = PairBuffer(); global_slots = SlotBuffer(); task_rows = []
    rows = all_positive = 0; upper = _cross_primitive_upper(device)
    score_max_asymmetry = 0.0; score_exact_symmetric = True; score_all_finite = True
    for task_name in loader.task_names:
        task_pairs = PairBuffer(); task_slots = SlotBuffer(); task_positive = task_row_count = 0
        length = loader._lengths[task_name]
        for start in range(0, length, EVALUATION_BATCH_SIZE):
            indices = np.arange(start, min(start + EVALUATION_BATCH_SIZE, length), dtype=np.int64)
            numpy_batch = loader._read(task_name, indices)
            batch = observable_numpy_batch_to_torch(numpy_batch, device=device)
            prediction = model(
                batch.range_valid,
                batch.relative_translation_current_sensor_m,
                batch.relative_yaw_current_sensor_deg,
            )
            assignments = match_primitives(prediction.primitive, batch.targets)
            targets = align_local_composition_slot_targets(batch.targets, assignments)
            observed = targets.endpoint_supervised
            validity = observed[:, :, None] & observed[:, None, :] & upper[None]
            target = (
                targets.labels[:, :, None] >= 0
            ) & (targets.labels[:, :, None] == targets.labels[:, None, :]) & validity
            positives = int(target.sum()); all_positive += positives; task_positive += positives
            score = structured_same_slot_pair_score(prediction)
            score_max_asymmetry = max(
                score_max_asymmetry,
                float(torch.max(torch.abs(score - score.transpose(1, 2))).cpu()),
            )
            score_exact_symmetric &= torch.equal(score, score.transpose(1, 2))
            score_all_finite &= bool(torch.isfinite(score).all())
            active_primitive = torch.sigmoid(prediction.primitive.existence_logits) >= existence_threshold
            active_endpoint = active_primitive.repeat_interleave(2, dim=1)
            eligible = validity & active_endpoint[:, :, None] & active_endpoint[:, None, :]
            overlap = targets.disconnected_overlap & ~target
            global_pairs.append(score[eligible], target[eligible], overlap[eligible])
            task_pairs.append(score[eligible], target[eligible], overlap[eligible])
            best_slot, endpoint_confidence = structured_endpoint_slot_confidence(prediction)
            global_slots.append(best_slot, endpoint_confidence, active_endpoint, targets.cluster_count)
            task_slots.append(best_slot, endpoint_confidence, active_endpoint, targets.cluster_count)
            count = len(indices); rows += count; task_row_count += count
        task_pair_metrics = task_pairs.finalize(total_positive=task_positive)
        task_safe_threshold = (
            float(task_pair_metrics["safe"]["threshold"])
            if task_pair_metrics["safe"]["available"] else None
        )
        task_rows.append({
            "task": task_name, "rows": task_row_count,
            "objective_true_pairs": task_positive,
            "pair_metrics": task_pair_metrics,
            "slot_diagnostics_at_task_safe": task_slots.finalize(threshold=task_safe_threshold),
        })
    if rows != EXPECTED_ROWS or all_positive != EXPECTED_OBSERVABLE_POSITIVES:
        raise RuntimeError(f"local composition-slot C07 population drift: {rows}/{all_positive}")
    pair_metrics = global_pairs.finalize(total_positive=all_positive)
    safe_threshold = (
        float(pair_metrics["safe"]["threshold"])
        if pair_metrics["safe"]["available"] else None
    )
    return {
        "rows": rows, "existence_threshold": existence_threshold,
        "objective_true_pairs": all_positive,
        "pair_metrics": pair_metrics,
        "slot_diagnostics_at_safe": global_slots.finalize(threshold=safe_threshold),
        "score_contract": {
            "maximum_asymmetry": score_max_asymmetry,
            "exact_symmetric": score_exact_symmetric,
            "all_finite": score_all_finite,
            "single_structured_refusal_threshold": True,
        },
        "tasks": task_rows,
    }


def _plot(summary: dict, output: Path) -> None:
    seeds = summary["c07"]["seeds"]; baseline = summary["c07"]["baseline_attachment_f1"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    labels = ["baseline", *[f"seed{row['seed']}" for row in seeds]]
    f1 = [baseline, *[row["metrics"]["pair_metrics"]["best_f1"]["f1"] for row in seeds]]
    colors = ("#777777", "#287271", "#d37524", "#5b5f97")
    axes[0].bar(labels, f1, color=colors)
    axes[0].axhline(baseline + MINIMUM_F1_GAIN, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("C07 observable attachment F1")
    safe_tp = [row["metrics"]["pair_metrics"]["safe"]["true_positive"] for row in seeds]
    axes[1].bar(labels[1:], safe_tp, color=colors[1:])
    axes[1].set_ylabel("true links at precision ≥ 0.98")
    predicted = [
        row["metrics"]["slot_diagnostics_at_safe"]["mean_predicted_nontrivial_clusters_per_row"]
        for row in seeds
    ]
    teacher = seeds[0]["metrics"]["slot_diagnostics_at_safe"]["mean_teacher_clusters_per_row"]
    axes[2].bar(labels[1:], predicted, color=colors[1:], label="predicted at safe threshold")
    axes[2].axhline(teacher, color="black", linestyle="--", label="Teacher")
    axes[2].set_ylabel("nontrivial composition clusters / row"); axes[2].legend(fontsize=8)
    fig.suptitle("Local composition-slot C07 gate — unseen topology")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--sensor-root", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--observability-root", required=True, type=Path)
    parser.add_argument("--baseline-c07-summary", required=True, type=Path)
    parser.add_argument("--existence-threshold-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("local composition-slot C07 evaluation requires CUDA")
    device = torch.device("cuda")
    loader = ObservablePrimitiveRelationBatchLoader(
        args.sensor_root / "c07", args.teacher_root / "c07",
        args.observability_root / "c07",
    )
    if len(loader) != EXPECTED_ROWS or len(loader.task_names) != EXPECTED_TASKS:
        raise RuntimeError("local composition-slot evaluator C07 loader drift")
    baseline = json.loads(args.baseline_c07_summary.read_text(encoding="utf-8"))
    if (
        baseline.get("rows") != EXPECTED_ROWS
        or baseline.get("scientific_pass") is not True
        or baseline.get("observable_attachment_target_positives") != EXPECTED_OBSERVABLE_POSITIVES
    ):
        raise RuntimeError("local composition-slot non-learning baseline drift")
    baseline_f1 = float(baseline["attachment"]["f1"])
    threshold_source = json.loads(args.existence_threshold_source.read_text(encoding="utf-8"))
    thresholds = {int(row["seed"]): float(row["existence_threshold"]) for row in threshold_source["seeds"]}
    if set(thresholds) != {0, 1, 2}:
        raise RuntimeError("local composition-slot existence threshold drift")
    seeds = []
    for seed in range(3):
        model, checkpoint = _load_model(
            args.models_root / f"seed{seed}/selected.pt", seed=seed, device=device,
        )
        metrics = evaluate_seed(model, loader, existence_threshold=thresholds[seed], device=device)
        pair = metrics["pair_metrics"]
        checks = {
            "attachment_f1_gain_ge_0p05": pair["best_f1"]["f1"] - baseline_f1 >= MINIMUM_F1_GAIN,
            "safe_precision_ge_0p98_nonzero": pair["safe"]["available"]
                and pair["safe"]["precision"] >= MINIMUM_SAFE_PRECISION
                and pair["safe"]["true_positive"] > 0,
            "score_finite_exact_symmetric": metrics["score_contract"]["all_finite"]
                and metrics["score_contract"]["exact_symmetric"],
            "single_threshold_structured_decode": metrics["score_contract"]["single_structured_refusal_threshold"],
            "frozen_backbone_contract": checkpoint["training_contract"]["frozen_observable_backbone"] is True,
        }
        record = {
            "seed": seed, "selected_epoch": int(checkpoint["epoch"]),
            "metrics": metrics, "checks": checks, "pass": all(checks.values()),
        }
        seeds.append(record); _write(output / f"c07_seed{seed}.json", record)
        _write(output / f"c07_seed{seed}_tasks.json", metrics["tasks"])
        print(json.dumps({
            "seed": seed, "checks": checks, "best_f1": pair["best_f1"],
            "safe": pair["safe"], "slot_diagnostics": metrics["slot_diagnostics_at_safe"],
        }), flush=True)
    passing = sum(row["pass"] for row in seeds); scientific_pass = passing >= 2
    summary = {
        "schema_version": "primitive_local_composition_slot_three_seed_evaluation_v1",
        "overall_status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "c07": {
            "baseline_attachment_f1": baseline_f1,
            "seeds": seeds, "passing_seeds": passing,
            "scientific_pass": scientific_pass,
        },
        "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        "graph_replays": 0, "mtare_worlds_read": 0,
        "decision": "ALLOW_C08_LOCAL_COMPOSITION_SLOT_ZERO_ADAPTATION_TRANSFER"
            if scientific_pass else "STOP_LOCAL_COMPOSITION_SLOT_BEFORE_C08_AND_GRAPH",
        "duration_seconds": time.monotonic() - started,
    }
    _write(output / "summary.json", summary)
    _plot(summary, output / "primitive_local_composition_slot_c07_comparison")
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
