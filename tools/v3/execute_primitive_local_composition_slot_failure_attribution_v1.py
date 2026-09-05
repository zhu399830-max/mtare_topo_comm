#!/usr/bin/env python3
"""Frozen C07-only factor attribution for three local composition-slot models."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mtare_topo.data.primitive_relation_observable_batches import ObservablePrimitiveRelationBatchLoader
from mtare_topo.evaluation.primitive_local_composition_slot_failure_attribution import (
    factorize_local_slot_confidence,
    local_slot_diagnostic_pair_scores,
)
from mtare_topo.evaluation.primitive_relation_sparse_port_failure_attribution import (
    exact_ranked_selection,
    score_quantiles,
)
from mtare_topo.representation.primitive_local_composition_slot_model import (
    COMPOSITION_SLOT_COUNT,
    DUSTBIN_INDEX,
    FrozenObservableLocalCompositionSlotNet,
    composition_slot_probabilities,
)
from mtare_topo.representation.primitive_local_composition_slot_training import (
    align_local_composition_slot_targets,
    match_local_composition_slots,
)
from mtare_topo.representation.primitive_relation_losses import match_primitives
from mtare_topo.representation.primitive_relation_observable_model import ObservableSparsePortRelationNet
from mtare_topo.representation.primitive_relation_observable_training import observable_numpy_batch_to_torch


EXPECTED_ROWS = 64_644
EXPECTED_TASKS = 30
EXPECTED_POSITIVES = 442_936
BATCH_SIZE = 128
MINIMUM_PRECISION = 0.98
PASS = "PASS_PRIMITIVE_LOCAL_COMPOSITION_SLOT_FAILURE_ATTRIBUTION_V1"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_model(path: Path, seed: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != "primitive_local_composition_slot_checkpoint_v1"
        or checkpoint.get("seed") != seed
        or checkpoint.get("training_contract", {}).get("composition_slot_count") != 32
    ):
        raise RuntimeError("local-slot attribution checkpoint identity drift")
    model = FrozenObservableLocalCompositionSlotNet(ObservableSparsePortRelationNet())
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def _upper(device: torch.device) -> torch.Tensor:
    endpoint = torch.arange(64, device=device)
    return (endpoint[:, None] // 2 != endpoint[None, :] // 2) & torch.triu(
        torch.ones(64, 64, dtype=torch.bool, device=device), diagonal=1,
    )


class FactorAccumulator:
    def __init__(self) -> None:
        self.counts = {group: 0 for group in ("all", "clustered", "dustbin")}
        self.values: dict[str, dict[str, dict[str, float | int]]] = {}

    def append(self, factors, masks: dict[str, torch.Tensor]) -> None:
        for name, value in factors.endpoint_scores().items():
            record = self.values.setdefault(name, {})
            for group, mask in masks.items():
                selected = value[mask]
                bucket = record.setdefault(group, {"count": 0, "zero": 0, "sum": 0.0, "maximum": 0.0})
                bucket["count"] += int(selected.numel())
                bucket["zero"] += int((selected == 0).sum())
                bucket["sum"] += float(selected.sum().cpu())
                if selected.numel():
                    bucket["maximum"] = max(float(bucket["maximum"]), float(selected.max().cpu()))
        for group, mask in masks.items():
            self.counts[group] += int(mask.sum())

    def finalize(self) -> dict:
        result = {}
        for name, groups in self.values.items():
            result[name] = {}
            for group, values in groups.items():
                count = int(values["count"])
                result[name][group] = {
                    "count": count,
                    "zero_count": int(values["zero"]),
                    "zero_fraction": float(values["zero"]) / max(count, 1),
                    "mean": float(values["sum"]) / max(count, 1),
                    "maximum": float(values["maximum"]),
                }
        return result


class PairAccumulator:
    def __init__(self) -> None:
        self.scores: dict[str, list[np.ndarray]] = {}
        self.target: list[np.ndarray] = []
        self.overlap: list[np.ndarray] = []

    def append(self, scores, target, overlap, eligible) -> None:
        for name, value in scores.items():
            self.scores.setdefault(name, []).append(
                value[eligible].detach().cpu().numpy().astype(np.float32, copy=False)
            )
        self.target.append(target[eligible].detach().cpu().numpy().astype(np.bool_, copy=False))
        self.overlap.append(overlap[eligible].detach().cpu().numpy().astype(np.bool_, copy=False))

    def finalize(self) -> dict:
        target = np.concatenate(self.target)
        overlap = np.concatenate(self.overlap)
        result = {}
        for name in sorted(self.scores):
            score = np.concatenate(self.scores[name])
            best = exact_ranked_selection(score, target, total_positive=EXPECTED_POSITIVES)
            safe = exact_ranked_selection(
                score, target, total_positive=EXPECTED_POSITIVES,
                minimum_precision=MINIMUM_PRECISION,
            )
            positive = target
            result[name] = {
                "best_f1": best,
                "safe": safe,
                "score_quantiles": score_quantiles(score, target),
                "nonzero_true_pairs": int(np.count_nonzero((score > 0) & positive)),
                "nonzero_false_pairs": int(np.count_nonzero((score > 0) & ~positive)),
                "overlap_fp_at_best": int(np.count_nonzero(
                    overlap & ~positive & (score >= float(best["threshold"]))
                )) if best["available"] else 0,
                "overlap_fp_at_safe": int(np.count_nonzero(
                    overlap & ~positive & (score >= float(safe["threshold"]))
                )) if safe["available"] else 0,
            }
            del score
        result["population"] = {
            "candidate_pairs": int(len(target)),
            "candidate_true_pairs": int(np.count_nonzero(target)),
            "objective_true_pairs": EXPECTED_POSITIVES,
            "overlap_hard_negative_pairs": int(np.count_nonzero(overlap & ~target)),
        }
        return result


def _mapped_endpoint_accuracy(prediction, targets, factors) -> dict[str, int]:
    mappings = match_local_composition_slots(prediction.composition, targets)
    expected = torch.full_like(targets.labels, DUSTBIN_INDEX)
    expected[~targets.endpoint_supervised] = -2
    for row, mapping in enumerate(mappings):
        for cluster, slot in enumerate(mapping.tolist()):
            expected[row, targets.labels[row] == cluster] = slot
    full_argmax = composition_slot_probabilities(prediction.composition).argmax(-1)
    clustered = targets.endpoint_supervised & (targets.labels >= 0)
    dustbin = targets.endpoint_supervised & (targets.labels == -1)
    return {
        "clustered_endpoints": int(clustered.sum()),
        "clustered_best_slot_correct": int((clustered & (factors.best_slot == expected)).sum()),
        "clustered_full_argmax_correct": int((clustered & (full_argmax == expected)).sum()),
        "clustered_dustbin_argmax": int((clustered & (full_argmax == DUSTBIN_INDEX)).sum()),
        "dustbin_endpoints": int(dustbin.sum()),
        "dustbin_argmax_correct": int((dustbin & (full_argmax == DUSTBIN_INDEX)).sum()),
    }


@torch.no_grad()
def evaluate_seed(model, loader, existence_threshold: float, device: torch.device) -> dict:
    upper = _upper(device); pair = PairAccumulator(); endpoint = FactorAccumulator()
    endpoint_counts = {key: 0 for key in (
        "clustered_endpoints", "clustered_best_slot_correct", "clustered_full_argmax_correct",
        "clustered_dustbin_argmax", "dustbin_endpoints", "dustbin_argmax_correct",
    )}
    rows = positives = 0; tasks = []
    for task_name in loader.task_names:
        task_rows = task_positive = task_candidate = task_same_true = task_same_false = 0
        task_joint_zero = task_clustered = 0
        length = loader._lengths[task_name]
        for start in range(0, length, BATCH_SIZE):
            indices = np.arange(start, min(start + BATCH_SIZE, length), dtype=np.int64)
            batch = observable_numpy_batch_to_torch(loader._read(task_name, indices), device=device)
            prediction = model(
                batch.range_valid, batch.relative_translation_current_sensor_m,
                batch.relative_yaw_current_sensor_deg,
            )
            assignments = match_primitives(prediction.primitive, batch.targets)
            targets = align_local_composition_slot_targets(batch.targets, assignments)
            observed = targets.endpoint_supervised
            validity = observed[:, :, None] & observed[:, None, :] & upper[None]
            target = (targets.labels[:, :, None] >= 0) & (
                targets.labels[:, :, None] == targets.labels[:, None, :]
            ) & validity
            active_primitive = torch.sigmoid(prediction.primitive.existence_logits) >= existence_threshold
            active = active_primitive.repeat_interleave(2, dim=1)
            eligible = validity & active[:, :, None] & active[:, None, :]
            overlap = targets.disconnected_overlap & ~target
            factors = factorize_local_slot_confidence(prediction)
            scores = local_slot_diagnostic_pair_scores(prediction)
            pair.append(scores, target, overlap, eligible)
            clustered_endpoint = observed & (targets.labels >= 0) & active
            dustbin_endpoint = observed & (targets.labels == -1) & active
            endpoint.append(factors, {
                "all": observed & active,
                "clustered": clustered_endpoint,
                "dustbin": dustbin_endpoint,
            })
            accuracy = _mapped_endpoint_accuracy(prediction, targets, factors)
            for key, value in accuracy.items(): endpoint_counts[key] += value
            same = factors.best_slot[:, :, None] == factors.best_slot[:, None, :]
            count = len(indices); value_positive = int(target.sum()); value_candidate = int(eligible.sum())
            rows += count; positives += value_positive
            task_rows += count; task_positive += value_positive; task_candidate += value_candidate
            task_same_true += int((same & target & eligible).sum())
            task_same_false += int((same & ~target & eligible).sum())
            task_joint_zero += int((factors.joint_margin[clustered_endpoint] == 0).sum())
            task_clustered += int(clustered_endpoint.sum())
        tasks.append({
            "task": task_name, "rows": task_rows, "objective_true_pairs": task_positive,
            "candidate_pairs": task_candidate,
            "true_pairs_with_same_best_slot": task_same_true,
            "false_pairs_with_same_best_slot": task_same_false,
            "clustered_endpoint_joint_margin_zero_fraction": task_joint_zero / max(task_clustered, 1),
        })
    if rows != EXPECTED_ROWS or positives != EXPECTED_POSITIVES:
        raise RuntimeError(f"local-slot attribution population drift: {rows}/{positives}")
    metrics = pair.finalize()
    clustered = endpoint_counts["clustered_endpoints"]
    dustbin = endpoint_counts["dustbin_endpoints"]
    endpoint_counts.update({
        "clustered_best_slot_accuracy_after_hungarian": endpoint_counts["clustered_best_slot_correct"] / max(clustered, 1),
        "clustered_full_argmax_accuracy_after_hungarian": endpoint_counts["clustered_full_argmax_correct"] / max(clustered, 1),
        "clustered_dustbin_argmax_fraction": endpoint_counts["clustered_dustbin_argmax"] / max(clustered, 1),
        "dustbin_argmax_accuracy": endpoint_counts["dustbin_argmax_correct"] / max(dustbin, 1),
    })
    return {
        "rows": rows, "objective_true_pairs": positives,
        "existence_threshold": existence_threshold,
        "endpoint_assignment": endpoint_counts,
        "endpoint_factor_statistics": endpoint.finalize(),
        "pair_factor_metrics": metrics,
        "tasks": tasks,
    }


def _same_selection(left: dict, right: dict) -> bool:
    keys = ("available", "threshold", "true_positive", "false_positive", "false_negative", "selected_pairs")
    return all(left.get(key) == right.get(key) for key in keys) and abs(left["f1"] - right["f1"]) <= 1e-15


def _plot(seeds: list[dict], output: Path) -> None:
    names = [
        "structured_deployed", "hard_slot_identity", "hard_slot_assignment_only",
        "hard_slot_without_margin", "soft_slot_affinity_no_presence",
        "soft_slot_relation_with_presence", "soft_slot_safe_legacy",
    ]
    labels = ["deployed", "hard identity", "assignment", "no margin", "soft affinity", "soft+presence", "legacy soft safe"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    x = np.arange(len(names)); width = .24
    for seed in seeds:
        values = [seed["metrics"]["pair_factor_metrics"][name]["best_f1"]["f1"] for name in names]
        safe = [seed["metrics"]["pair_factor_metrics"][name]["safe"]["true_positive"] for name in names]
        axes[0].bar(x + (seed["seed"] - 1) * width, values, width, label=f"seed{seed['seed']}")
        axes[1].bar(x + (seed["seed"] - 1) * width, safe, width, label=f"seed{seed['seed']}")
    axes[0].set_ylabel("C07 attachment best F1")
    axes[1].set_ylabel("true links at precision ≥ 0.98")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=28, ha="right"); axis.legend()
    fig.suptitle("Frozen local-slot failure attribution on C07")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--observability-root", type=Path, required=True)
    parser.add_argument("--source-evaluation-summary", type=Path, required=True)
    parser.add_argument("--existence-threshold-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if not torch.cuda.is_available(): raise RuntimeError("local-slot attribution requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(False)
    device = torch.device("cuda")
    loader = ObservablePrimitiveRelationBatchLoader(
        args.sensor_root / "c07", args.teacher_root / "c07", args.observability_root / "c07",
    )
    if len(loader) != EXPECTED_ROWS or len(loader.task_names) != EXPECTED_TASKS:
        raise RuntimeError("local-slot attribution loader drift")
    source = json.loads(args.source_evaluation_summary.read_text(encoding="utf-8"))
    threshold_source = json.loads(args.existence_threshold_source.read_text(encoding="utf-8"))
    thresholds = {int(row["seed"]): float(row["existence_threshold"]) for row in threshold_source["seeds"]}
    seeds = []
    for seed in range(3):
        model, checkpoint = _load_model(args.models_root / f"seed{seed}/selected.pt", seed, device)
        metrics = evaluate_seed(model, loader, thresholds[seed], device)
        source_record = source["c07"]["seeds"][seed]["metrics"]["pair_metrics"]
        deployed = metrics["pair_factor_metrics"]["structured_deployed"]
        reproduction = {
            "best_f1": _same_selection(deployed["best_f1"], source_record["best_f1"]),
            "safe": _same_selection(deployed["safe"], source_record["safe"]),
            "candidate_pairs": metrics["pair_factor_metrics"]["population"]["candidate_pairs"] == source_record["candidate_pairs"],
            "candidate_true_pairs": metrics["pair_factor_metrics"]["population"]["candidate_true_pairs"] == source_record["candidate_true_pairs"],
        }
        record = {"seed": seed, "selected_epoch": int(checkpoint["epoch"]), "metrics": metrics, "source_reproduction": reproduction}
        if not all(reproduction.values()): raise RuntimeError(f"seed{seed} source metric reproduction drift: {reproduction}")
        seeds.append(record); _write(output / f"seed{seed}.json", record); _write(output / f"seed{seed}_tasks.json", metrics["tasks"])
        del model; gc.collect(); torch.cuda.empty_cache()
        print(json.dumps({
            "seed": seed, "assignment": metrics["endpoint_assignment"],
            "deployed": deployed, "hard_identity": metrics["pair_factor_metrics"]["hard_slot_identity"],
            "soft_affinity": metrics["pair_factor_metrics"]["soft_slot_affinity_no_presence"],
        }), flush=True)

    def safe_seed_count(name: str) -> int:
        return sum(
            row["metrics"]["pair_factor_metrics"][name]["safe"]["available"]
            and row["metrics"]["pair_factor_metrics"][name]["safe"]["true_positive"] > 0
            for row in seeds
        )

    hard_identity_safe = safe_seed_count("hard_slot_identity")
    soft_affinity_safe = safe_seed_count("soft_slot_affinity_no_presence")
    factor_safe_counts = {
        name: safe_seed_count(name)
        for name in seeds[0]["metrics"]["pair_factor_metrics"]
        if name != "population"
    }
    if hard_identity_safe >= 2:
        diagnosis = "MULTIPLICATIVE_CONFIDENCE_COLLAPSE_WITH_SAFE_HARD_SLOT_IDENTITY"
        recommendation = "Keep the exchangeable slot identity and redesign a single calibrated endpoint acceptance score in a new readiness; do not change the graph gate."
    elif soft_affinity_safe >= 2:
        diagnosis = "HARD_ARGMAX_SLOT_FRAGMENTATION_WITH_SOFT_RELATION_SIGNAL"
        recommendation = "Replace hard argmax equality with a separately trained transitive clustering objective; the C07 diagnostic soft score is not deployable as the new method."
    else:
        diagnosis = "SLOT_IDENTITY_AND_CONFIDENCE_NOT_CROSS_SEED_SAFE"
        recommendation = "Stop the current local-slot assignment route; redesign the relation supervision around explicit pair/cluster contrast without relying on arbitrary slot identity."
    summary = {
        "schema_version": "primitive_local_composition_slot_failure_attribution_v1",
        "overall_status": PASS, "scientific_pass": True,
        "attribution_diagnosis": diagnosis, "recommendation": recommendation,
        "hard_identity_safe_seeds": hard_identity_safe,
        "soft_affinity_safe_seeds": soft_affinity_safe,
        "safe_seed_counts_by_factor": factor_safe_counts,
        "seeds": seeds,
        "checks": {
            "three_source_metrics_exactly_reproduced": all(all(row["source_reproduction"].values()) for row in seeds),
            "complete_c07_population": all(row["metrics"]["rows"] == EXPECTED_ROWS for row in seeds),
            "objective_positive_population": all(row["metrics"]["objective_true_pairs"] == EXPECTED_POSITIVES for row in seeds),
            "all_registered_factors_present": len(factor_safe_counts) == 18,
            "zero_optimizer_steps": True, "zero_c08_graph_mtare": True,
        },
        "optimizer_steps": 0, "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        "graph_replays": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    }
    if not all(summary["checks"].values()): raise RuntimeError(f"attribution evidence incomplete: {summary['checks']}")
    _write(output / "summary.json", summary)
    _plot(seeds, output / "primitive_local_composition_slot_failure_attribution")
    print(json.dumps({
        "overall_status": PASS, "diagnosis": diagnosis,
        "safe_seed_counts_by_factor": factor_safe_counts,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
