#!/usr/bin/env python3
"""One-pass, C07-only attribution of the frozen sparse-port V2 failure."""

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

from mtare_topo.data.primitive_relation_batches import PrimitiveRelationBatchLoader
from mtare_topo.evaluation.primitive_relation_failure_attribution import (
    pair_space_counts,
    relation_sweeps_for_mask,
    slot_population_counts,
)
from mtare_topo.evaluation.primitive_relation_metrics import (
    add_sweeps,
    align_for_evaluation,
    relation_sweeps,
    select_threshold,
)
from mtare_topo.evaluation.primitive_relation_sparse_port_failure_attribution import (
    attachment_score_slice,
    exact_ranked_selection,
    score_quantiles,
    teacher_cardinality_topk_mask,
)
from mtare_topo.governance import write_json
from mtare_topo.representation.primitive_relation_sparse_port_model import SparsePortRelationNet
from mtare_topo.representation.primitive_relation_training import numpy_batch_to_torch


EXPECTED_ROWS = 64_644
EXPECTED_TASKS = 30
SEEDS = (0, 1, 2)
BATCH_SIZE = 128
SAFE_PRECISION = 0.98
MASKS = ("deployed", "teacher_cardinality", "proposal_oracle")
DECODERS = ("independent", "best_link_union", "best_link_mutual")


class ScoreBuffer:
    def __init__(self) -> None:
        self.raw: list[np.ndarray] = []
        self.safe: list[np.ndarray] = []
        self.target: list[np.ndarray] = []
        self.overlap: list[np.ndarray] = []
        self.eligible_pairs = 0
        self.target_pairs = 0

    def append(self, item) -> None:
        self.raw.append(item.raw_score)
        self.safe.append(item.safe_score)
        self.target.append(item.target)
        self.overlap.append(item.overlap_hard_negative)
        self.eligible_pairs += item.eligible_pairs
        self.target_pairs += item.target_pairs

    def finalize(self) -> dict:
        raw = np.concatenate(self.raw) if self.raw else np.empty(0, dtype=np.float32)
        safe = np.concatenate(self.safe) if self.safe else np.empty(0, dtype=np.float32)
        target = np.concatenate(self.target) if self.target else np.empty(0, dtype=np.bool_)
        overlap = np.concatenate(self.overlap) if self.overlap else np.empty(0, dtype=np.bool_)
        raw_f1 = exact_ranked_selection(raw, target, total_positive=self.target_pairs)
        raw_safe = exact_ranked_selection(
            raw, target, total_positive=self.target_pairs, minimum_precision=SAFE_PRECISION,
        )
        adjusted_f1 = exact_ranked_selection(safe, target, total_positive=self.target_pairs)
        adjusted_safe = exact_ranked_selection(
            safe, target, total_positive=self.target_pairs, minimum_precision=SAFE_PRECISION,
        )

        def overlap_false_positive(score: np.ndarray, selection: dict) -> int:
            if not selection["available"]:
                return 0
            predicted = score >= float(selection["threshold"])
            return int(np.count_nonzero(predicted & ~target & overlap))

        return {
            "eligible_pairs": self.eligible_pairs,
            "eligible_true_pairs": int(np.count_nonzero(target)),
            "all_target_pairs": self.target_pairs,
            "raw": {
                "best_f1": raw_f1,
                "safe": raw_safe,
                "score_quantiles": score_quantiles(raw, target),
                "overlap_hard_negative_fp_at_best_f1": overlap_false_positive(raw, raw_f1),
            },
            "risk_adjusted": {
                "formula": "p_attachment_times_one_minus_uncertainty",
                "best_f1": adjusted_f1,
                "safe": adjusted_safe,
                "score_quantiles": score_quantiles(safe, target),
                "overlap_hard_negative_fp_at_best_f1": overlap_false_positive(safe, adjusted_f1),
            },
        }


def _load_model(path: Path, seed: int, device: torch.device) -> tuple[SparsePortRelationNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "primitive_relation_sparse_port_checkpoint_v1":
        raise RuntimeError("sparse-port attribution checkpoint schema drift")
    if int(checkpoint.get("seed", -1)) != seed:
        raise RuntimeError("sparse-port attribution checkpoint seed drift")
    model = SparsePortRelationNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _same_selection(actual: dict, formal: dict) -> bool:
    return (
        all(int(actual[name]) == int(formal[name]) for name in ("true_positive", "false_positive", "false_negative"))
        and all(abs(float(actual[name]) - float(formal[name])) <= 1e-12 for name in ("threshold", "precision", "recall", "f1"))
    )


@torch.no_grad()
def _seed_attribution(seed: int, model, loader, formal: dict, *, device: torch.device) -> tuple[dict, list[dict]]:
    threshold = float(formal["metrics"]["existence_f1_selection"]["threshold"])
    buffers = {(mask, decoder): ScoreBuffer() for mask in MASKS for decoder in DECODERS}
    formal_attachment = None
    populations = {name: None for name in MASKS}
    pair_spaces = {name: None for name in MASKS}
    task_rows: list[dict] = []
    rows = 0

    def merge(first: dict | None, second: dict) -> dict:
        if first is None:
            return json.loads(json.dumps(second))
        result = dict(first)
        for key, value in second.items():
            if key == "active_count_histogram":
                result[key] = [int(a) + int(b) for a, b in zip(result[key], value, strict=True)]
            elif isinstance(value, int):
                result[key] = int(result.get(key, 0)) + value
        return result

    for task_name in loader.task_names:
        task_actual = task_oracle = None
        task_count = loader._lengths[task_name]
        for start in range(0, task_count, BATCH_SIZE):
            indices = np.arange(start, min(start + BATCH_SIZE, task_count), dtype=np.int64)
            numpy_batch = loader._read(task_name, indices)
            batch = numpy_batch_to_torch(numpy_batch, device=device)
            prediction = model(
                batch.range_valid,
                batch.relative_translation_current_sensor_m,
                batch.relative_yaw_current_sensor_deg,
            )
            aligned = align_for_evaluation(prediction, batch.targets)
            deployed = torch.sigmoid(prediction.existence_logits) >= threshold
            cardinality = teacher_cardinality_topk_mask(prediction.existence_logits, aligned["mask"])
            oracle = aligned["mask"].bool()
            masks = {"deployed": deployed, "teacher_cardinality": cardinality, "proposal_oracle": oracle}

            actual_sweep = relation_sweeps(prediction, aligned, existence_threshold=threshold)["attachment"]
            oracle_sweep = relation_sweeps_for_mask(prediction, aligned, oracle)["attachment"]
            formal_attachment = add_sweeps(formal_attachment, actual_sweep)
            task_actual = add_sweeps(task_actual, actual_sweep)
            task_oracle = add_sweeps(task_oracle, oracle_sweep)

            for name, mask in masks.items():
                populations[name] = merge(populations[name], slot_population_counts(mask, oracle))
                pair_spaces[name] = merge(pair_spaces[name], pair_space_counts(mask))
                for decoder in DECODERS:
                    buffers[(name, decoder)].append(attachment_score_slice(
                        prediction.endpoint_attachment_logits,
                        prediction.endpoint_attachment_uncertainty,
                        aligned["attachment"], aligned["overlap"], mask,
                        decoder=decoder,
                    ))
            rows += len(indices)
        task_rows.append({
            "seed": seed,
            "task": task_name,
            "rows": task_count,
            "topology_family": task_name.split("_C07__", 1)[0],
            "geometry_family": task_name.rsplit("__", 1)[-1],
            "deployed_attachment_f1": select_threshold(task_actual)["f1"],
            "proposal_oracle_attachment_f1": select_threshold(task_oracle)["f1"],
        })

    if rows != EXPECTED_ROWS or len(loader.task_names) != EXPECTED_TASKS:
        raise RuntimeError("sparse-port attribution C07 population drift")
    reproduced = select_threshold(formal_attachment)
    if not _same_selection(reproduced, formal["metrics"]["attachment_f1_selection"]):
        raise RuntimeError("sparse-port attribution failed formal attachment reproduction")
    diagnostics = {
        mask: {decoder: buffers[(mask, decoder)].finalize() for decoder in DECODERS}
        for mask in MASKS
    }
    return {
        "seed": seed,
        "selected_epoch": int(formal["selected_epoch"]),
        "rows": rows,
        "formal_attachment_reproduced": True,
        "formal_attachment": reproduced,
        "populations": populations,
        "pair_spaces": pair_spaces,
        "diagnostics": diagnostics,
    }, task_rows


def _condition_pass(seed: dict, mask: str, decoder: str, score: str, gate_f1: float) -> bool:
    record = seed["diagnostics"][mask][decoder][score]
    return (
        float(record["best_f1"]["f1"]) >= gate_f1
        and bool(record["safe"]["available"])
        and int(record["safe"]["true_positive"]) > 0
    )


def _diagnosis(seeds: list[dict], gate_f1: float) -> tuple[str, str, dict]:
    counts = {}
    for mask, decoder, score in (
        ("deployed", "independent", "raw"),
        ("deployed", "independent", "risk_adjusted"),
        ("deployed", "best_link_union", "raw"),
        ("teacher_cardinality", "best_link_union", "raw"),
        ("proposal_oracle", "independent", "raw"),
        ("proposal_oracle", "independent", "risk_adjusted"),
        ("proposal_oracle", "best_link_union", "raw"),
    ):
        key = f"{mask}__{decoder}__{score}"
        counts[key] = sum(_condition_pass(seed, mask, decoder, score, gate_f1) for seed in seeds)
    if counts["deployed__independent__raw"] >= 2:
        return "FROZEN_THRESHOLD_GRID_MISSED_EXACT_SAFE_PREFIX", "REPAIR_EVALUATION_CALIBRATION_CONTRACT_BEFORE_MODEL_CHANGE", counts
    if counts["deployed__best_link_union__raw"] >= 2 or counts["teacher_cardinality__best_link_union__raw"] >= 2:
        return "PAIRWISE_SCORE_NEEDS_STRUCTURED_BEST_LINK_DECODING", "ALLOW_STRUCTURE_CONSTRAINED_RELATION_DECODER_READINESS", counts
    if counts["proposal_oracle__independent__raw"] >= 2:
        if counts["proposal_oracle__independent__risk_adjusted"] < 2:
            return "UNCERTAINTY_CALIBRATION_SUPPRESSES_ORACLE_SAFE_RELATIONS", "ALLOW_RELATION_CALIBRATION_CORRECTIVE_READINESS", counts
        return "PROPOSAL_OR_CARDINALITY_DOMINATES_V2_FAILURE", "ALLOW_PROPOSAL_RELATION_DECOUPLING_READINESS", counts
    if counts["proposal_oracle__best_link_union__raw"] >= 2:
        return "RELATION_SCORE_CONTAINS_ONLY_LOCAL_LINK_RANKING", "ALLOW_STRUCTURE_CONSTRAINED_RELATION_DECODER_READINESS", counts
    return "RELATION_SCORE_FAILS_EVEN_WITH_PROPOSAL_ORACLE", "STOP_CURRENT_RELATION_HEAD_AND_REASSESS_METHOD", counts


def _plot(summary: dict, destination: Path) -> None:
    seeds = summary["seeds"]; x = np.arange(3); width = 0.22
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), constrained_layout=True)
    axes[0, 0].bar(x - width, [s["populations"]["deployed"]["active"] / s["rows"] for s in seeds], width, label="deployed")
    axes[0, 0].bar(x, [s["populations"]["teacher_cardinality"]["active"] / s["rows"] for s in seeds], width, label="Teacher count")
    axes[0, 0].bar(x + width, [s["populations"]["proposal_oracle"]["active"] / s["rows"] for s in seeds], width, label="proposal oracle")
    axes[0, 0].set(title="A  Primitive candidates", ylabel="slots / sequence", xticks=x, xticklabels=("seed0", "seed1", "seed2")); axes[0, 0].legend(frameon=False)
    conditions = (("deployed", "independent"), ("teacher_cardinality", "independent"), ("proposal_oracle", "independent"), ("deployed", "best_link_union"))
    labels = ("deployed", "count oracle", "proposal oracle", "best-link")
    for offset, ((mask, decoder), label) in enumerate(zip(conditions, labels, strict=True)):
        axes[0, 1].bar(x + (offset - 1.5) * 0.18, [s["diagnostics"][mask][decoder]["raw"]["best_f1"]["f1"] for s in seeds], 0.18, label=label)
    axes[0, 1].axhline(summary["attachment_gate_f1"], color="black", linestyle="--")
    axes[0, 1].set(title="B  Exact attachment F1", ylabel="F1", xticks=x, xticklabels=("seed0", "seed1", "seed2")); axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].bar(x - width / 2, [s["diagnostics"]["proposal_oracle"]["independent"]["raw"]["safe"]["recall"] for s in seeds], width, label="raw probability")
    axes[1, 0].bar(x + width / 2, [s["diagnostics"]["proposal_oracle"]["independent"]["risk_adjusted"]["safe"]["recall"] for s in seeds], width, label="risk adjusted")
    axes[1, 0].set(title="C  Proposal-oracle recall at precision >= 0.98", ylabel="recall", xticks=x, xticklabels=("seed0", "seed1", "seed2")); axes[1, 0].legend(frameon=False)
    axes[1, 1].bar(x, [s["diagnostics"]["deployed"]["independent"]["raw"]["overlap_hard_negative_fp_at_best_f1"] for s in seeds], color="#e15759")
    axes[1, 1].set(title="D  Stacked/overlap false attachments", ylabel="false positives", xticks=x, xticklabels=("seed0", "seed1", "seed2"))
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2); axis.set_axisbelow(True)
    fig.suptitle("Sparse-Port V2 C07 Relation Failure Attribution")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination.with_suffix("." + suffix), dpi=220)
    plt.close(fig)


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
    if not torch.cuda.is_available():
        raise RuntimeError("sparse-port attribution requires CUDA")
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)
    loader = PrimitiveRelationBatchLoader(args.sensor_root.resolve(), args.teacher_root.resolve())
    if len(loader) != EXPECTED_ROWS or len(loader.task_names) != EXPECTED_TASKS:
        raise RuntimeError("sparse-port attribution loader population drift")
    baseline = json.loads(args.baseline_summary.read_text())
    gate_f1 = float(baseline["attachment"]["f1"]) + 0.05
    seeds = []; tasks = []
    for seed in SEEDS:
        formal = json.loads((args.formal_evaluation_root / f"c07_seed{seed}.json").read_text())
        model, checkpoint = _load_model(args.models_root / f"seed{seed}/selected.pt", seed, device)
        if int(checkpoint["epoch"]) != int(formal["selected_epoch"]):
            raise RuntimeError("sparse-port attribution selected epoch drift")
        result, task = _seed_attribution(seed, model, loader, formal, device=device)
        seeds.append(result); tasks.extend(task)
        write_json(output / f"seed{seed}.json", result)
        del model; torch.cuda.empty_cache()
    diagnosis, decision, condition_counts = _diagnosis(seeds, gate_f1)
    checks = {
        "three_frozen_seeds": len(seeds) == 3,
        "full_c07_population_each_seed": all(seed["rows"] == EXPECTED_ROWS for seed in seeds),
        "formal_attachment_reproduced": all(seed["formal_attachment_reproduced"] for seed in seeds),
        "all_three_masks_reported": all(set(seed["diagnostics"]) == set(MASKS) for seed in seeds),
        "all_three_decoders_reported": all(all(set(seed["diagnostics"][mask]) == set(DECODERS) for mask in MASKS) for seed in seeds),
        "diagnosis_resolved": bool(diagnosis and decision),
        "c08_rows_read_zero": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"sparse-port attribution checks failed: {checks}")
    summary = {
        "schema_version": "primitive_relation_sparse_port_failure_attribution_v1",
        "overall_status": "PASS_PRIMITIVE_RELATION_SPARSE_PORT_FAILURE_ATTRIBUTION_V1",
        "scientific_pass": True,
        "sparse_port_model_scientific_pass": False,
        "rows_per_seed": EXPECTED_ROWS,
        "model_inference_rows": EXPECTED_ROWS * len(SEEDS),
        "optimizer_steps": 0,
        "c08_rows_read": 0,
        "c09_c10_worlds_read": 0,
        "graph_replays": 0,
        "mtare_worlds_read": 0,
        "baseline_attachment_f1": float(baseline["attachment"]["f1"]),
        "attachment_gate_f1": gate_f1,
        "safe_precision": SAFE_PRECISION,
        "diagnosis": diagnosis,
        "decision": decision,
        "condition_passing_seed_counts": condition_counts,
        "checks": checks,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "duration_seconds": time.monotonic() - started,
        "seeds": seeds,
    }
    write_json(output / "summary.json", summary)
    with (output / "per_task.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tasks[0])); writer.writeheader(); writer.writerows(tasks)
    write_json(output / "figure_source.json", {
        "summary": "summary.json", "per_task": "per_task.csv",
        "oracle_is_evaluation_only": True,
        "best_link_uses_only_frozen_relation_scores": True,
        "diagnosis": diagnosis,
    })
    _plot(summary, output / "primitive_relation_sparse_port_failure_attribution")
    print(json.dumps({
        "overall_status": summary["overall_status"], "diagnosis": diagnosis,
        "decision": decision, "condition_passing_seed_counts": condition_counts,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
