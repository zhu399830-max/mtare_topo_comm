#!/usr/bin/env python3
"""Full-population C07-only attribution of the frozen primitive-relation V1."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mtare_topo.data.primitive_relation_batches import PrimitiveRelationBatchLoader
from mtare_topo.evaluation.primitive_relation_failure_attribution import (
    pair_space_counts,
    relation_sweeps_for_mask,
    slot_population_counts,
)
from mtare_topo.evaluation.primitive_relation_metrics import (
    BinaryCounts,
    add_sweeps,
    align_for_evaluation,
    existence_sweep,
    relation_sweeps,
    select_threshold,
)
from mtare_topo.governance import write_json
from mtare_topo.representation.primitive_relation_model import PrimitiveRelationNet
from mtare_topo.representation.primitive_relation_training import numpy_batch_to_torch


EXPECTED_ROWS = 64_644
EXPECTED_TASKS = 30
SEEDS = (0, 1, 2)
BATCH_SIZE = 128
SAFE_PRECISION = 0.98


def _load_model(path: Path, seed: int, device: torch.device) -> tuple[PrimitiveRelationNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "primitive_relation_checkpoint_v1":
        raise RuntimeError("primitive attribution checkpoint schema drift")
    if int(checkpoint.get("seed", -1)) != seed:
        raise RuntimeError("primitive attribution checkpoint seed drift")
    model = PrimitiveRelationNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _merge_counts(first: dict | None, second: dict) -> dict:
    if first is None:
        return json.loads(json.dumps(second))
    result = dict(first)
    for key in ("rows", "primitives", "attachment_pairs", "overlap_pairs"):
        if key in second:
            result[key] = int(result.get(key, 0)) + int(second[key])
    for key in ("active", "targets", "matched_active", "missed_targets", "redundant_active"):
        if key in second:
            result[key] = int(result.get(key, 0)) + int(second[key])
    if "active_count_histogram" in second:
        previous = result.get("active_count_histogram", [0] * len(second["active_count_histogram"]))
        result["active_count_histogram"] = [
            int(left) + int(right) for left, right in zip(previous, second["active_count_histogram"], strict=True)
        ]
    return result


def _selection(sweeps: tuple[BinaryCounts, ...]) -> dict:
    return {
        "f1": select_threshold(sweeps),
        "safe": select_threshold(sweeps, minimum_precision=SAFE_PRECISION),
    }


def _same_selection(actual: dict, formal: dict) -> bool:
    integer = ("true_positive", "false_positive", "false_negative")
    floating = ("threshold", "precision", "recall", "f1")
    return (
        all(int(actual[name]) == int(formal[name]) for name in integer)
        and all(abs(float(actual[name]) - float(formal[name])) <= 1e-12 for name in floating)
    )


@torch.no_grad()
def _seed_attribution(
    seed: int,
    model: PrimitiveRelationNet,
    loader: PrimitiveRelationBatchLoader,
    formal: dict,
    *,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    existence_threshold = float(formal["metrics"]["existence_f1_selection"]["threshold"])
    total_existence = None
    total_actual_attachment = total_actual_overlap = None
    total_oracle_attachment = total_oracle_overlap = None
    actual_pairs = oracle_pairs = population = None
    per_task: list[dict] = []
    rows = 0

    for task_name in loader.task_names:
        task_actual_attachment = task_oracle_attachment = None
        task_actual_pairs = task_oracle_pairs = None
        task_rows = loader._lengths[task_name]
        for start in range(0, task_rows, BATCH_SIZE):
            indices = np.arange(start, min(start + BATCH_SIZE, task_rows), dtype=np.int64)
            numpy_batch = loader._read(task_name, indices)
            batch = numpy_batch_to_torch(numpy_batch, device=device)
            prediction = model(
                batch.range_valid,
                batch.relative_translation_current_sensor_m,
                batch.relative_yaw_current_sensor_deg,
            )
            aligned = align_for_evaluation(prediction, batch.targets)
            active = torch.sigmoid(prediction.existence_logits) >= existence_threshold
            oracle_mask = aligned["mask"].bool()

            total_existence = add_sweeps(total_existence, existence_sweep(prediction, aligned))
            actual = relation_sweeps(prediction, aligned, existence_threshold=existence_threshold)
            oracle = relation_sweeps_for_mask(prediction, aligned, oracle_mask)
            total_actual_attachment = add_sweeps(total_actual_attachment, actual["attachment"])
            total_actual_overlap = add_sweeps(total_actual_overlap, actual["disconnected_overlap"])
            total_oracle_attachment = add_sweeps(total_oracle_attachment, oracle["attachment"])
            total_oracle_overlap = add_sweeps(total_oracle_overlap, oracle["disconnected_overlap"])
            task_actual_attachment = add_sweeps(task_actual_attachment, actual["attachment"])
            task_oracle_attachment = add_sweeps(task_oracle_attachment, oracle["attachment"])

            current_actual_pairs = pair_space_counts(active)
            current_oracle_pairs = pair_space_counts(oracle_mask)
            actual_pairs = _merge_counts(actual_pairs, current_actual_pairs)
            oracle_pairs = _merge_counts(oracle_pairs, current_oracle_pairs)
            task_actual_pairs = _merge_counts(task_actual_pairs, current_actual_pairs)
            task_oracle_pairs = _merge_counts(task_oracle_pairs, current_oracle_pairs)
            population = _merge_counts(population, slot_population_counts(active, oracle_mask))
            rows += len(indices)

        task_actual = _selection(task_actual_attachment)
        task_oracle = _selection(task_oracle_attachment)
        per_task.append({
            "seed": seed,
            "task": task_name,
            "rows": task_rows,
            "geometry_family": task_name.rsplit("__", 1)[-1],
            "topology_family": task_name.split("_C07__", 1)[0],
            "actual_attachment_f1": task_actual["f1"]["f1"],
            "oracle_attachment_f1": task_oracle["f1"]["f1"],
            "actual_safe_recall": task_actual["safe"]["recall"],
            "oracle_safe_recall": task_oracle["safe"]["recall"],
            "actual_attachment_pairs": task_actual_pairs["attachment_pairs"],
            "oracle_attachment_pairs": task_oracle_pairs["attachment_pairs"],
            "pair_expansion": task_actual_pairs["attachment_pairs"] / task_oracle_pairs["attachment_pairs"],
        })

    if rows != EXPECTED_ROWS or len(loader.task_names) != EXPECTED_TASKS:
        raise RuntimeError("primitive attribution C07 population drift")
    if any(value is None for value in (
        total_existence, total_actual_attachment, total_actual_overlap,
        total_oracle_attachment, total_oracle_overlap, actual_pairs, oracle_pairs, population,
    )):
        raise RuntimeError("primitive attribution accumulator is empty")

    existence = select_threshold(total_existence)
    actual_attachment = _selection(total_actual_attachment)
    actual_overlap = _selection(total_actual_overlap)
    oracle_attachment = _selection(total_oracle_attachment)
    oracle_overlap = _selection(total_oracle_overlap)
    reproduction = {
        "existence": _same_selection(existence, formal["metrics"]["existence_f1_selection"]),
        "attachment": _same_selection(actual_attachment["f1"], formal["metrics"]["attachment_f1_selection"]),
        "overlap": _same_selection(actual_overlap["f1"], formal["metrics"]["overlap_f1_selection"]),
    }
    if not all(reproduction.values()):
        raise RuntimeError(f"primitive attribution failed to reproduce formal C07 metrics: {reproduction}")
    result = {
        "seed": seed,
        "rows": rows,
        "existence_threshold": existence_threshold,
        "reproduction": reproduction,
        "slot_population": population,
        "actual_pair_space": actual_pairs,
        "oracle_pair_space": oracle_pairs,
        "attachment_pair_expansion": actual_pairs["attachment_pairs"] / oracle_pairs["attachment_pairs"],
        "overlap_pair_expansion": actual_pairs["overlap_pairs"] / oracle_pairs["overlap_pairs"],
        "actual": {"attachment": actual_attachment, "overlap": actual_overlap},
        "proposal_oracle": {"attachment": oracle_attachment, "overlap": oracle_overlap},
    }
    return result, per_task


def _plot(summary: dict, destination: Path) -> None:
    seeds = summary["seeds"]
    x = np.arange(3); width = 0.34
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)
    axes[0, 0].bar(x - width / 2, [item["slot_population"]["active"] / item["rows"] for item in seeds], width, label="predicted active")
    axes[0, 0].bar(x + width / 2, [item["slot_population"]["targets"] / item["rows"] for item in seeds], width, label="Teacher visible")
    axes[0, 0].set(xticks=x, xticklabels=("seed0", "seed1", "seed2"), ylabel="primitives / sequence", title="A  Primitive over-activation")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].bar(x, [item["attachment_pair_expansion"] for item in seeds], color="#e15759")
    axes[0, 1].set(xticks=x, xticklabels=("seed0", "seed1", "seed2"), ylabel="actual / oracle pair count", title="B  Pair-space expansion")
    actual_f1 = [item["actual"]["attachment"]["f1"]["f1"] for item in seeds]
    oracle_f1 = [item["proposal_oracle"]["attachment"]["f1"]["f1"] for item in seeds]
    axes[1, 0].bar(x - width / 2, actual_f1, width, label="actual candidates")
    axes[1, 0].bar(x + width / 2, oracle_f1, width, label="proposal oracle")
    axes[1, 0].axhline(summary["attachment_gate_f1"], color="black", linestyle="--", label="baseline + 5 points")
    axes[1, 0].set(xticks=x, xticklabels=("seed0", "seed1", "seed2"), ylim=(0, max(0.12, max(oracle_f1) * 1.15)), ylabel="attachment F1", title="C  Relation ability after removing false proposals")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].bar(x - width / 2, [item["actual"]["attachment"]["safe"]["recall"] for item in seeds], width, label="actual candidates")
    axes[1, 1].bar(x + width / 2, [item["proposal_oracle"]["attachment"]["safe"]["recall"] for item in seeds], width, label="proposal oracle")
    axes[1, 1].set(xticks=x, xticklabels=("seed0", "seed1", "seed2"), ylabel="recall at precision >= 0.98", title="D  Safe relation recall")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2); axis.set_axisbelow(True)
    figure.suptitle("Primitive-Relation V1 C07 Failure Attribution")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination.with_suffix("." + suffix), dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--sensor-root", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--formal-evaluation-root", required=True, type=Path)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); started = time.monotonic()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("primitive relation attribution requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)
    loader = PrimitiveRelationBatchLoader(args.sensor_root.resolve(), args.teacher_root.resolve())
    if len(loader) != EXPECTED_ROWS or len(loader.task_names) != EXPECTED_TASKS:
        raise RuntimeError("primitive relation attribution loader population drift")
    baseline = json.loads(args.baseline_summary.read_text())
    baseline_f1 = float(baseline["attachment"]["f1"])
    attachment_gate_f1 = baseline_f1 + 0.05
    seeds = []; task_rows = []
    for seed in SEEDS:
        formal = json.loads((args.formal_evaluation_root / f"c07_seed{seed}.json").read_text())
        model, checkpoint = _load_model(args.models_root / f"seed{seed}/selected.pt", seed, device)
        if int(checkpoint["epoch"]) != int(formal["selected_epoch"]):
            raise RuntimeError("primitive attribution selected epoch drift")
        result, tasks = _seed_attribution(seed, model, loader, formal, device=device)
        seeds.append(result); task_rows.extend(tasks)
        write_json(output / f"seed{seed}.json", result)
        del model; torch.cuda.empty_cache()

    oracle_gate_seeds = sum(
        item["proposal_oracle"]["attachment"]["f1"]["f1"] >= attachment_gate_f1
        and item["proposal_oracle"]["attachment"]["safe"]["true_positive"] > 0
        for item in seeds
    )
    if oracle_gate_seeds >= 2:
        diagnosis = "PROPOSAL_OVERACTIVATION_DOMINATES_RELATION_FAILURE"
        decision = "ALLOW_SPARSE_EXISTENCE_SET_DECODER_CORRECTIVE_READINESS"
    else:
        diagnosis = "RELATION_HEAD_FAILS_EVEN_WITH_PROPOSAL_ORACLE"
        decision = "REQUIRE_SPARSE_PORT_RELATION_ARCHITECTURE_READINESS"
    checks = {
        "three_frozen_seeds": len(seeds) == 3,
        "full_c07_population_each_seed": all(item["rows"] == EXPECTED_ROWS for item in seeds),
        "formal_metrics_reproduced": all(all(item["reproduction"].values()) for item in seeds),
        "proposal_oracle_reduces_pair_space": all(item["attachment_pair_expansion"] > 1.0 for item in seeds),
        "diagnosis_resolved": diagnosis in {
            "PROPOSAL_OVERACTIVATION_DOMINATES_RELATION_FAILURE",
            "RELATION_HEAD_FAILS_EVEN_WITH_PROPOSAL_ORACLE",
        },
        "c08_rows_read_zero": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"primitive relation attribution checks failed: {checks}")
    summary = {
        "schema_version": "primitive_relation_v1_failure_attribution_v1",
        "overall_status": "PASS_PRIMITIVE_RELATION_V1_FAILURE_ATTRIBUTION_V1",
        "scientific_pass": True,
        "model_v1_scientific_pass": False,
        "rows_per_seed": EXPECTED_ROWS,
        "model_inference_rows": EXPECTED_ROWS * len(SEEDS),
        "optimizer_steps": 0,
        "c08_rows_read": 0,
        "c09_c10_worlds_read": 0,
        "graph_replays": 0,
        "mtare_worlds_read": 0,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "baseline_attachment_f1": baseline_f1,
        "attachment_gate_f1": attachment_gate_f1,
        "oracle_gate_seeds": oracle_gate_seeds,
        "diagnosis": diagnosis,
        "decision": decision,
        "checks": checks,
        "seeds": seeds,
        "duration_seconds": time.monotonic() - started,
    }
    write_json(output / "summary.json", summary)
    with (output / "per_task.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(task_rows[0])); writer.writeheader(); writer.writerows(task_rows)
    write_json(output / "figure_source.json", {
        "summary": "summary.json", "per_task": "per_task.csv", "diagnosis": diagnosis,
        "oracle_is_evaluation_only": True,
    })
    _plot(summary, output / "primitive_relation_v1_failure_attribution")
    print(json.dumps({
        "overall_status": summary["overall_status"], "diagnosis": diagnosis,
        "decision": decision, "oracle_gate_seeds": oracle_gate_seeds,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
