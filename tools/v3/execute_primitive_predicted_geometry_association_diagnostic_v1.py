#!/usr/bin/env python3
"""Compare C07 predicted-endpoint geometry with learned pair scores for 3 seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mtare_topo.data.primitive_relation_observable_batches import (
    ObservablePrimitiveRelationBatchLoader,
)
from mtare_topo.evaluation.primitive_predicted_geometry_association import (
    predicted_geometry_association_slice,
)
from mtare_topo.evaluation.primitive_relation_metrics import align_for_evaluation
from mtare_topo.evaluation.primitive_relation_sparse_port_failure_attribution import (
    exact_ranked_selection,
)
from mtare_topo.governance import write_json
from mtare_topo.representation.primitive_relation_observable_model import (
    ObservableSparsePortRelationNet,
)
from mtare_topo.representation.primitive_relation_observable_training import (
    observable_numpy_batch_to_torch,
)


EXPECTED_ROWS = 64_644
EXPECTED_TASKS = 30
EXPECTED_POSITIVES = 442_936
SEEDS = (0, 1, 2)
BATCH_SIZE = 128
SAFE_PRECISION = .98


class Buffer:
    def __init__(self) -> None:
        self.geometry = []
        self.learned = []
        self.target = []
        self.overlap = []
        self.total_positive = 0

    def append(self, value) -> None:
        self.geometry.append(value.geometry_score)
        self.learned.append(value.learned_pair_score)
        self.target.append(value.target)
        self.overlap.append(value.overlap_hard_negative)
        self.total_positive += int(value.all_observable_target_pairs)

    def finalize(self, fixed_geometry_threshold: float) -> dict:
        geometry = np.concatenate(self.geometry)
        learned = np.concatenate(self.learned)
        target = np.concatenate(self.target)
        overlap = np.concatenate(self.overlap)
        result = {
            "eligible_pairs": len(target),
            "eligible_true_pairs": int(np.count_nonzero(target)),
            "all_observable_target_pairs": self.total_positive,
        }
        for name, score in (("predicted_geometry", geometry), ("learned_pair", learned)):
            best = exact_ranked_selection(score, target, total_positive=self.total_positive)
            safe = exact_ranked_selection(
                score, target, total_positive=self.total_positive,
                minimum_precision=SAFE_PRECISION,
            )
            selected = score >= float(best["threshold"]) if best["available"] else np.zeros_like(target)
            safe_selected = score >= float(safe["threshold"]) if safe["available"] else np.zeros_like(target)
            result[name] = {
                "best_f1": best, "safe": safe,
                "overlap_hard_negative_fp_at_best_f1": int(np.count_nonzero(selected & ~target & overlap)),
                "overlap_hard_negative_fp_at_safe": int(np.count_nonzero(safe_selected & ~target & overlap)),
            }
        predicted = geometry >= float(fixed_geometry_threshold)
        tp = int(np.count_nonzero(predicted & target))
        fp = int(np.count_nonzero(predicted & ~target))
        fn = self.total_positive - tp
        result["teacher_fit_fixed_geometry_threshold"] = {
            "threshold_score": float(fixed_geometry_threshold),
            "threshold_distance_m": float(-fixed_geometry_threshold),
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / self.total_positive if self.total_positive else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "overlap_hard_negative_false_positive": int(np.count_nonzero(predicted & ~target & overlap)),
        }
        return result


def _load_model(path: Path, seed: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != "primitive_relation_observable_checkpoint_v1"
        or int(checkpoint.get("seed", -1)) != seed
        or checkpoint.get("training_contract", {}).get("teacher_attachment_validity")
        != "dual_endpoint_observed"
    ):
        raise RuntimeError("predicted-geometry checkpoint identity drift")
    model = ObservableSparsePortRelationNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def _seed(seed, model, loader, formal, fixed_geometry_threshold, device):
    existence_threshold = float(formal["metrics"]["existence_f1_selection"]["threshold"])
    buffers = {name: Buffer() for name in ("deployed", "proposal_oracle")}
    per_task = []
    rows = 0
    for task_name in loader.task_names:
        task_fixed = {name: {"tp": 0, "fp": 0, "positive": 0} for name in buffers}
        task_rows = loader._lengths[task_name]
        for start in range(0, task_rows, BATCH_SIZE):
            indices = np.arange(start, min(start + BATCH_SIZE, task_rows), dtype=np.int64)
            numpy_batch = loader._read(task_name, indices)
            batch = observable_numpy_batch_to_torch(numpy_batch, device=device)
            prediction = model(
                batch.range_valid,
                batch.relative_translation_current_sensor_m,
                batch.relative_yaw_current_sensor_deg,
            )
            aligned = align_for_evaluation(prediction, batch.targets)
            masks = {
                "deployed": torch.sigmoid(prediction.existence_logits) >= existence_threshold,
                "proposal_oracle": aligned["mask"].bool(),
            }
            for name, mask in masks.items():
                value = predicted_geometry_association_slice(prediction, aligned, mask)
                buffers[name].append(value)
                selected = value.geometry_score >= float(fixed_geometry_threshold)
                task_fixed[name]["tp"] += int(np.count_nonzero(selected & value.target))
                task_fixed[name]["fp"] += int(np.count_nonzero(selected & ~value.target))
                task_fixed[name]["positive"] += int(value.all_observable_target_pairs)
            rows += len(indices)
        row = {"seed": seed, "task": task_name, "rows": task_rows}
        for name, counts in task_fixed.items():
            tp, fp, positive = counts["tp"], counts["fp"], counts["positive"]
            row[f"{name}_fixed_tp"] = tp
            row[f"{name}_fixed_fp"] = fp
            row[f"{name}_fixed_fn"] = positive - tp
            row[f"{name}_fixed_precision"] = tp / (tp + fp) if tp + fp else 0.0
            row[f"{name}_fixed_recall"] = tp / positive if positive else 0.0
        per_task.append(row)
    if rows != EXPECTED_ROWS or len(loader.task_names) != EXPECTED_TASKS:
        raise RuntimeError("predicted-geometry C07 population drift")
    result = {
        "seed": seed, "rows": rows, "tasks": len(loader.task_names),
        "existence_threshold": existence_threshold,
        "masks": {
            name: value.finalize(fixed_geometry_threshold)
            for name, value in buffers.items()
        },
    }
    if any(
        result["masks"][name]["all_observable_target_pairs"] != EXPECTED_POSITIVES
        for name in result["masks"]
    ):
        raise RuntimeError("predicted-geometry observable positive population drift")
    return result, per_task


def _diagnosis(seed_results):
    deployed = [
        value["masks"]["deployed"]["predicted_geometry"]["safe"]["true_positive"] > 0
        for value in seed_results
    ]
    oracle = [
        value["masks"]["proposal_oracle"]["predicted_geometry"]["safe"]["true_positive"] > 0
        for value in seed_results
    ]
    deployed_fixed = [
        value["masks"]["deployed"]["teacher_fit_fixed_geometry_threshold"]["precision"] >= .98
        and value["masks"]["deployed"]["teacher_fit_fixed_geometry_threshold"]["true_positive"] > 0
        for value in seed_results
    ]
    if sum(deployed) >= 2:
        return {
            "diagnosis": "PREDICTED_GEOMETRY_COMPOSITION_ALREADY_HAS_SAFE_DEPLOYED_CONNECTIONS",
            "decision": "STOP_COMPOSITION_ANCHOR_HEAD_AS_UNNECESSARY_AND_QUALIFY_GEOMETRY_COMPOSITION_BASELINE",
            "deployed_geometry_safe_passing_seeds": sum(deployed),
            "proposal_oracle_geometry_safe_passing_seeds": sum(oracle),
            "teacher_fit_fixed_geometry_passing_seeds": sum(deployed_fixed),
        }
    if sum(oracle) >= 2:
        return {
            "diagnosis": "PREDICTED_GEOMETRY_RELATION_IS_SAFE_ONLY_WITH_PROPOSAL_ORACLE",
            "decision": "STOP_ANCHOR_ONLY_HEAD_AND_REASSESS_PRIMITIVE_PROPOSAL_CONSOLIDATION",
            "deployed_geometry_safe_passing_seeds": sum(deployed),
            "proposal_oracle_geometry_safe_passing_seeds": sum(oracle),
            "teacher_fit_fixed_geometry_passing_seeds": sum(deployed_fixed),
        }
    return {
        "diagnosis": "PREDICTED_ENDPOINT_GEOMETRY_IS_NOT_SAFE_EVEN_WITH_PROPOSAL_ORACLE",
        "decision": "ALLOW_OE_COMPOSITION_ANCHOR_RESIDUAL_UNCERTAINTY_MODEL_READINESS",
        "deployed_geometry_safe_passing_seeds": sum(deployed),
        "proposal_oracle_geometry_safe_passing_seeds": sum(oracle),
        "teacher_fit_fixed_geometry_passing_seeds": sum(deployed_fixed),
    }


def _plot(summary, output):
    seeds = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    width = .18
    series = []
    for mask in ("deployed", "proposal_oracle"):
        for score in ("predicted_geometry", "learned_pair"):
            values = [
                row["masks"][mask][score]["best_f1"]["f1"]
                for row in summary["seeds"]
            ]
            series.append((f"{mask}\n{score}", values))
    for index, (label, values) in enumerate(series):
        axes[0].bar(seeds + (index - 1.5) * width, values, width=width, label=label)
    axes[0].set_xticks(seeds, ["seed0", "seed1", "seed2"])
    axes[0].set_ylabel("best C07 attachment F1"); axes[0].legend(fontsize=7)
    masks = ("deployed", "proposal_oracle")
    for index, mask in enumerate(masks):
        values = [
            row["masks"][mask]["predicted_geometry"]["safe"]["true_positive"]
            for row in summary["seeds"]
        ]
        axes[1].bar(seeds + (index - .5) * .3, values, width=.3, label=mask)
    axes[1].set_xticks(seeds, ["seed0", "seed1", "seed2"])
    axes[1].set_ylabel("true connections at precision ≥ 0.98")
    axes[1].legend(fontsize=8)
    fig.suptitle("Are learned composition anchors necessary? Frozen C07 predicted geometry")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"primitive_predicted_geometry_association_diagnostic.{suffix}", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--sensor-root", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument("--formal-evaluation-root", required=True, type=Path)
    parser.add_argument("--anchor-readiness-summary", required=True, type=Path)
    parser.add_argument("--attribution-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    readiness = json.loads(args.anchor_readiness_summary.read_text(encoding="utf-8"))
    threshold_distance = float(
        readiness["axis_endpoint_baseline"]["fit_safe_selection"]["threshold"] * -1.0
    )
    fixed_geometry_threshold = -threshold_distance
    loader = ObservablePrimitiveRelationBatchLoader(
        args.sensor_root.resolve(), args.teacher_root.resolve(), args.sidecar_root.resolve(),
    )
    if len(loader) != EXPECTED_ROWS or len(loader.task_names) != EXPECTED_TASKS:
        raise RuntimeError("predicted-geometry loader population drift")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    seed_results = []; per_task = []; peak_allocated = peak_reserved = 0
    for seed in SEEDS:
        model = _load_model(args.models_root / f"seed{seed}/selected.pt", seed, device)
        formal = json.loads((args.formal_evaluation_root / f"c07_seed{seed}.json").read_text())
        result, task = _seed(
            seed, model, loader, formal, fixed_geometry_threshold, device,
        )
        seed_results.append(result); per_task.extend(task)
        peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))
        peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved()))
        del model; torch.cuda.empty_cache()
    attribution = json.loads(args.attribution_summary.read_text(encoding="utf-8"))["attribution"]
    attribution_by_seed = {int(value["seed"]): value for value in attribution["seeds"]}
    learned_pair_reproduced = True
    for result in seed_results:
        source = attribution_by_seed[int(result["seed"])]
        for mask in ("deployed", "proposal_oracle"):
            actual = result["masks"][mask]["learned_pair"]["best_f1"]
            expected = source["diagnostics"][f"{mask}__independent"]["raw"]["best_f1"]
            learned_pair_reproduced &= all(
                int(actual[name]) == int(expected[name])
                for name in ("true_positive", "false_positive", "false_negative")
            ) and all(
                abs(float(actual[name]) - float(expected[name])) <= 1e-12
                for name in ("threshold", "precision", "recall", "f1")
            )
    if not learned_pair_reproduced:
        raise RuntimeError("predicted-geometry diagnostic failed sealed learned-pair reproduction")
    decision = _diagnosis(seed_results)
    summary = {
        "schema_version": "primitive_predicted_geometry_association_diagnostic_v1",
        "seeds": seed_results, **decision,
        "sealed_learned_pair_metrics_reproduced": learned_pair_reproduced,
        "teacher_fit_distance_threshold_m": threshold_distance,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "model_forward_rows": EXPECTED_ROWS * len(SEEDS),
        "optimizer_steps": 0, "c07_rows_per_seed": EXPECTED_ROWS,
        "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        "graph_replays": 0, "mtare_worlds_read": 0,
    }
    write_json(output / "summary.json", summary)
    for result in seed_results:
        write_json(output / f"seed{result['seed']}.json", result)
    write_json(output / "per_task.json", {"rows": per_task})
    with (output / "per_task.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_task[0]))
        writer.writeheader(); writer.writerows(per_task)
    write_json(output / "figure_source.json", {
        "schema_version": "primitive_predicted_geometry_association_figure_source_v1",
        "question": "Are learned anchor residuals necessary after predicted primitive geometry?",
        "seeds": seed_results, **decision,
    })
    _plot(summary, output)
    print(json.dumps({**decision, "model_forward_rows": summary["model_forward_rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
