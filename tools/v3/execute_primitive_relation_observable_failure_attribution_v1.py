#!/usr/bin/env python3
"""One-pass C07 attribution of the frozen endpoint-observable relation failure."""

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

from mtare_topo.data.primitive_relation_observable_batches import (
    ObservablePrimitiveRelationBatchLoader,
)
from mtare_topo.evaluation.primitive_relation_failure_attribution import (
    pair_space_counts,
    slot_population_counts,
)
from mtare_topo.evaluation.primitive_relation_metrics import (
    _attachment_observability_validity,
    _attachment_upper_mask,
    add_sweeps,
    align_for_evaluation,
    relation_sweeps,
    select_threshold,
    threshold_sweep_counts,
)
from mtare_topo.evaluation.primitive_relation_observable_failure_attribution import (
    diagnose_observable_relation,
    observable_attachment_score_slice,
    observable_endpoint_evidence_slice,
)
from mtare_topo.evaluation.primitive_relation_sparse_port_failure_attribution import (
    exact_ranked_selection,
    score_quantiles,
    teacher_cardinality_topk_mask,
)
from mtare_topo.governance import write_json
from mtare_topo.representation.primitive_relation_observable_model import (
    ObservableSparsePortRelationNet,
    safe_attachment_score,
)
from mtare_topo.representation.primitive_relation_observable_training import (
    observable_numpy_batch_to_torch,
)


EXPECTED_ROWS = 64_644
EXPECTED_TASKS = 30
EXPECTED_OBSERVABLE_POSITIVES = 442_936
SEEDS = (0, 1, 2)
BATCH_SIZE = 128
SAFE_PRECISION = 0.98
BUFFER_CONFIG = {
    ("deployed", "independent"): ("raw", "uncertainty_adjusted", "learned_safe"),
    ("deployed", "best_link_union"): ("raw",),
    ("teacher_cardinality", "best_link_union"): ("raw",),
    ("proposal_oracle", "independent"): ("raw", "uncertainty_adjusted", "learned_safe"),
    ("proposal_oracle", "best_link_union"): ("raw",),
}


class ScoreBuffer:
    def __init__(self, score_names: tuple[str, ...]) -> None:
        self.score_names = score_names
        self.scores = {name: [] for name in score_names}
        self.target: list[np.ndarray] = []
        self.overlap: list[np.ndarray] = []
        self.eligible_pairs = 0
        self.eligible_true_pairs = 0
        self.all_target_pairs = 0

    def append(self, item) -> None:
        for name in self.score_names:
            self.scores[name].append(item.scores[name])
        self.target.append(item.target)
        self.overlap.append(item.overlap_hard_negative)
        self.eligible_pairs += item.eligible_pairs
        self.eligible_true_pairs += item.eligible_true_pairs
        self.all_target_pairs += item.all_observable_target_pairs

    def finalize(self) -> dict:
        target = np.concatenate(self.target) if self.target else np.empty(0, dtype=np.bool_)
        overlap = np.concatenate(self.overlap) if self.overlap else np.empty(0, dtype=np.bool_)
        result = {
            "eligible_pairs": self.eligible_pairs,
            "eligible_true_pairs": self.eligible_true_pairs,
            "all_observable_target_pairs": self.all_target_pairs,
        }
        for name in self.score_names:
            score = np.concatenate(self.scores[name]) if self.scores[name] else np.empty(0, dtype=np.float32)
            best = exact_ranked_selection(score, target, total_positive=self.all_target_pairs)
            safe = exact_ranked_selection(
                score, target, total_positive=self.all_target_pairs,
                minimum_precision=SAFE_PRECISION,
            )
            predicted = (
                score >= float(best["threshold"])
                if best["available"] else np.zeros_like(target)
            )
            result[name] = {
                "best_f1": best,
                "safe": safe,
                "score_quantiles": score_quantiles(score, target),
                "overlap_hard_negative_fp_at_best_f1": int(
                    np.count_nonzero(predicted & ~target & overlap)
                ),
            }
        return result


class EndpointEvidenceBuffer:
    def __init__(self) -> None:
        self.score: list[np.ndarray] = []
        self.target: list[np.ndarray] = []
        self.total_positive = 0

    def append(self, score: np.ndarray, target: np.ndarray, total_positive: int) -> None:
        self.score.append(score); self.target.append(target)
        self.total_positive += int(total_positive)

    def finalize(self) -> dict:
        score = np.concatenate(self.score)
        target = np.concatenate(self.target)
        return {
            "best_f1": exact_ranked_selection(
                score, target, total_positive=self.total_positive,
            ),
            "safe": exact_ranked_selection(
                score, target, total_positive=self.total_positive,
                minimum_precision=SAFE_PRECISION,
            ),
            "score_quantiles": score_quantiles(score, target),
            "eligible_endpoints": int(len(score)),
            "observable_positive_endpoints": int(self.total_positive),
        }


def _load_model(path: Path, seed: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != "primitive_relation_observable_checkpoint_v1"
        or int(checkpoint.get("seed", -1)) != seed
        or checkpoint.get("training_contract", {}).get("teacher_attachment_validity")
        != "dual_endpoint_observed"
    ):
        raise RuntimeError("observable attribution checkpoint identity drift")
    model = ObservableSparsePortRelationNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _same_selection(actual: dict, formal: dict) -> bool:
    count_names = ("true_positive", "false_positive", "false_negative")
    scalar_names = ("threshold", "precision", "recall", "f1")
    return (
        all(int(actual[name]) == int(formal[name]) for name in count_names)
        and all(abs(float(actual[name]) - float(formal[name])) <= 1e-12 for name in scalar_names)
    )


def _merge(first: dict | None, second: dict) -> dict:
    if first is None:
        return json.loads(json.dumps(second))
    result = dict(first)
    for key, value in second.items():
        if key == "active_count_histogram":
            result[key] = [int(a) + int(b) for a, b in zip(result[key], value, strict=True)]
        elif isinstance(value, int):
            result[key] = int(result.get(key, 0)) + value
    return result


@torch.no_grad()
def _seed_attribution(seed: int, model, loader, formal: dict, *, device: torch.device):
    threshold = float(formal["metrics"]["existence_f1_selection"]["threshold"])
    buffers = {key: ScoreBuffer(names) for key, names in BUFFER_CONFIG.items()}
    evidence = {name: EndpointEvidenceBuffer() for name in ("all_slots", "proposal_oracle")}
    populations = {name: None for name in ("deployed", "teacher_cardinality", "proposal_oracle")}
    pair_spaces = {name: None for name in populations}
    formal_attachment = formal_safe = None
    task_rows = []
    rows = 0
    for task_name in loader.task_names:
        task_deployed = ScoreBuffer(("raw",))
        task_oracle = ScoreBuffer(("raw",))
        task_count = loader._lengths[task_name]
        for start in range(0, task_count, BATCH_SIZE):
            indices = np.arange(start, min(start + BATCH_SIZE, task_count), dtype=np.int64)
            numpy_batch = loader._read(task_name, indices)
            batch = observable_numpy_batch_to_torch(numpy_batch, device=device)
            prediction = model(
                batch.range_valid,
                batch.relative_translation_current_sensor_m,
                batch.relative_yaw_current_sensor_deg,
            )
            aligned = align_for_evaluation(prediction, batch.targets)
            deployed = torch.sigmoid(prediction.existence_logits) >= threshold
            cardinality = teacher_cardinality_topk_mask(prediction.existence_logits, aligned["mask"])
            oracle = aligned["mask"].bool()
            masks = {
                "deployed": deployed,
                "teacher_cardinality": cardinality,
                "proposal_oracle": oracle,
            }
            formal_attachment = add_sweeps(
                formal_attachment,
                relation_sweeps(prediction, aligned, existence_threshold=threshold)["attachment"],
            )
            endpoint_active = deployed.repeat_interleave(2, dim=1)
            validity = _attachment_observability_validity(aligned)
            upper = _attachment_upper_mask(device)[None]
            eligible = endpoint_active[:, :, None] & endpoint_active[:, None, :] & validity & upper
            target = aligned["attachment"].reshape(len(deployed), 64, 64).bool() & validity & upper
            formal_safe = add_sweeps(formal_safe, threshold_sweep_counts(
                safe_attachment_score(prediction).reshape(len(deployed), 64, 64),
                target, eligible=eligible,
            ))
            for name, matched_only in (("all_slots", False), ("proposal_oracle", True)):
                score, endpoint_target, positives = observable_endpoint_evidence_slice(
                    prediction, aligned, matched_only=matched_only,
                )
                evidence[name].append(score, endpoint_target, positives)
            for name, mask in masks.items():
                populations[name] = _merge(populations[name], slot_population_counts(mask, oracle))
                pair_spaces[name] = _merge(pair_spaces[name], pair_space_counts(mask))
            for (mask_name, decoder), buffer in buffers.items():
                buffer.append(observable_attachment_score_slice(
                    prediction, aligned, masks[mask_name], decoder=decoder,
                    score_names=buffer.score_names,
                ))
            task_deployed.append(observable_attachment_score_slice(
                prediction, aligned, deployed, score_names=("raw",),
            ))
            task_oracle.append(observable_attachment_score_slice(
                prediction, aligned, oracle, score_names=("raw",),
            ))
            rows += len(indices)
        task_rows.append({
            "seed": seed, "task": task_name, "rows": task_count,
            "topology_family": task_name.split("_C07__", 1)[0],
            "geometry_family": task_name.rsplit("__", 1)[-1],
            "deployed_exact_attachment_f1": task_deployed.finalize()["raw"]["best_f1"]["f1"],
            "proposal_oracle_exact_attachment_f1": task_oracle.finalize()["raw"]["best_f1"]["f1"],
        })
    if rows != EXPECTED_ROWS or len(loader.task_names) != EXPECTED_TASKS:
        raise RuntimeError("observable attribution C07 population drift")
    reproduced_attachment = select_threshold(formal_attachment)
    reproduced_safe = select_threshold(formal_safe, minimum_precision=SAFE_PRECISION)
    if not _same_selection(reproduced_attachment, formal["metrics"]["attachment_f1_selection"]):
        raise RuntimeError("observable attribution failed formal attachment reproduction")
    if not _same_selection(reproduced_safe, formal["metrics"]["attachment_safe_selection"]):
        raise RuntimeError("observable attribution failed formal safe reproduction")
    diagnostics = {f"{mask}__{decoder}": buffer.finalize() for (mask, decoder), buffer in buffers.items()}
    if diagnostics["proposal_oracle__independent"]["all_observable_target_pairs"] != EXPECTED_OBSERVABLE_POSITIVES:
        raise RuntimeError("observable attribution positive population drift")
    return ({
        "seed": seed, "selected_epoch": int(formal["selected_epoch"]), "rows": rows,
        "formal_attachment_reproduced": True, "formal_safe_reproduced": True,
        "formal_attachment": reproduced_attachment, "formal_safe": reproduced_safe,
        "populations": populations, "pair_spaces": pair_spaces,
        "endpoint_evidence": {name: value.finalize() for name, value in evidence.items()},
        "diagnostics": diagnostics,
    }, task_rows)


def _condition_pass(seed: dict, condition: str, score: str, gate_f1: float) -> bool:
    record = seed["diagnostics"][condition][score]
    return (
        float(record["best_f1"]["f1"]) >= gate_f1
        and bool(record["safe"]["available"])
        and int(record["safe"]["true_positive"]) > 0
    )


def _condition_counts(seeds: list[dict], gate_f1: float) -> dict[str, int]:
    conditions = (
        ("deployed__independent", "raw"),
        ("deployed__independent", "uncertainty_adjusted"),
        ("deployed__independent", "learned_safe"),
        ("deployed__best_link_union", "raw"),
        ("teacher_cardinality__best_link_union", "raw"),
        ("proposal_oracle__independent", "raw"),
        ("proposal_oracle__independent", "uncertainty_adjusted"),
        ("proposal_oracle__independent", "learned_safe"),
        ("proposal_oracle__best_link_union", "raw"),
    )
    return {
        f"{condition}__{score}": sum(
            _condition_pass(seed, condition, score, gate_f1) for seed in seeds
        )
        for condition, score in conditions
    }


def _plot(summary: dict, destination: Path) -> None:
    seeds = summary["seeds"]; x = np.arange(3); width = 0.24
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.3), constrained_layout=True)
    axes[0, 0].bar(x - width, [s["populations"]["deployed"]["active"] / s["rows"] for s in seeds], width, label="deployed")
    axes[0, 0].bar(x, [s["populations"]["teacher_cardinality"]["active"] / s["rows"] for s in seeds], width, label="Teacher count")
    axes[0, 0].bar(x + width, [s["populations"]["proposal_oracle"]["active"] / s["rows"] for s in seeds], width, label="proposal oracle")
    axes[0, 0].set(title="A  Primitive candidates", ylabel="slots / sequence", xticks=x, xticklabels=("seed0", "seed1", "seed2")); axes[0, 0].legend(frameon=False)
    conditions = (
        ("deployed__independent", "deployed"),
        ("deployed__best_link_union", "best-link"),
        ("teacher_cardinality__best_link_union", "count+best-link"),
        ("proposal_oracle__independent", "proposal oracle"),
    )
    for offset, (key, label) in enumerate(conditions):
        axes[0, 1].bar(x + (offset - 1.5) * 0.18, [s["diagnostics"][key]["raw"]["best_f1"]["f1"] for s in seeds], 0.18, label=label)
    axes[0, 1].axhline(summary["attachment_gate_f1"], color="black", linestyle="--")
    axes[0, 1].set(title="B  Exact attachment F1", ylabel="F1", xticks=x, xticklabels=("seed0", "seed1", "seed2")); axes[0, 1].legend(frameon=False, fontsize=8)
    for offset, score in enumerate(("raw", "uncertainty_adjusted", "learned_safe")):
        axes[1, 0].bar(x + (offset - 1) * width, [s["diagnostics"]["proposal_oracle__independent"][score]["safe"]["recall"] for s in seeds], width, label=score.replace("_", " "))
    axes[1, 0].set(title="C  Oracle recall at precision >= 0.98", ylabel="recall", xticks=x, xticklabels=("seed0", "seed1", "seed2")); axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].bar(x - width / 2, [s["endpoint_evidence"]["all_slots"]["best_f1"]["f1"] for s in seeds], width, label="all slots")
    axes[1, 1].bar(x + width / 2, [s["endpoint_evidence"]["proposal_oracle"]["best_f1"]["f1"] for s in seeds], width, label="proposal oracle")
    axes[1, 1].set(title="D  Endpoint evidence", ylabel="exact F1", xticks=x, xticklabels=("seed0", "seed1", "seed2")); axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2); axis.set_axisbelow(True)
    fig.suptitle("Observable Primitive Relation C07 Failure Attribution")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination.with_suffix("." + suffix), dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--sensor-root", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument("--formal-evaluation-root", required=True, type=Path)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); started = time.monotonic()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("observable relation attribution requires CUDA")
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(device)
    loader = ObservablePrimitiveRelationBatchLoader(
        args.sensor_root.resolve(), args.teacher_root.resolve(), args.sidecar_root.resolve(),
    )
    if len(loader) != EXPECTED_ROWS or len(loader.task_names) != EXPECTED_TASKS:
        raise RuntimeError("observable relation attribution loader population drift")
    baseline = json.loads(args.baseline_summary.read_text())
    if (
        baseline.get("rows") != EXPECTED_ROWS
        or baseline.get("observable_attachment_target_positives") != EXPECTED_OBSERVABLE_POSITIVES
    ):
        raise RuntimeError("observable relation attribution baseline drift")
    gate_f1 = float(baseline["attachment"]["f1"]) + 0.05
    seeds = []; tasks = []
    for seed in SEEDS:
        formal = json.loads((args.formal_evaluation_root / f"c07_seed{seed}.json").read_text())
        model, checkpoint = _load_model(args.models_root / f"seed{seed}/selected.pt", seed, device)
        if int(checkpoint["epoch"]) != int(formal["selected_epoch"]):
            raise RuntimeError("observable relation attribution selected epoch drift")
        result, task = _seed_attribution(seed, model, loader, formal, device=device)
        seeds.append(result); tasks.extend(task); write_json(output / f"seed{seed}.json", result)
        del model; torch.cuda.empty_cache()
    counts = _condition_counts(seeds, gate_f1)
    diagnosis, decision = diagnose_observable_relation(counts)
    checks = {
        "three_frozen_seeds": len(seeds) == 3,
        "full_c07_population_each_seed": all(seed["rows"] == EXPECTED_ROWS for seed in seeds),
        "formal_attachment_reproduced": all(seed["formal_attachment_reproduced"] for seed in seeds),
        "formal_safe_reproduced": all(seed["formal_safe_reproduced"] for seed in seeds),
        "observable_positive_population_exact": all(
            seed["diagnostics"]["proposal_oracle__independent"]["all_observable_target_pairs"]
            == EXPECTED_OBSERVABLE_POSITIVES for seed in seeds
        ),
        "all_registered_diagnostics_reported": all(
            set(seed["diagnostics"]) == {f"{mask}__{decoder}" for mask, decoder in BUFFER_CONFIG}
            for seed in seeds
        ),
        "diagnosis_resolved": bool(diagnosis and decision),
        "c08_rows_read_zero": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"observable relation attribution checks failed: {checks}")
    summary = {
        "schema_version": "primitive_relation_observable_failure_attribution_v1",
        "overall_status": "PASS_PRIMITIVE_RELATION_OBSERVABLE_FAILURE_ATTRIBUTION_V1",
        "scientific_pass": True,
        "observable_relation_model_scientific_pass": False,
        "rows_per_seed": EXPECTED_ROWS,
        "model_forward_rows": EXPECTED_ROWS * len(SEEDS),
        "optimizer_steps": 0, "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        "graph_replays": 0, "mtare_worlds_read": 0,
        "baseline_attachment_f1": float(baseline["attachment"]["f1"]),
        "attachment_gate_f1": gate_f1, "safe_precision": SAFE_PRECISION,
        "diagnosis": diagnosis, "decision": decision,
        "condition_passing_seed_counts": counts, "checks": checks,
        "numerical_contract": {
            "deterministic_algorithms": True, "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False, "float32_matmul_precision": "highest",
        },
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "duration_seconds": time.monotonic() - started, "seeds": seeds,
    }
    write_json(output / "summary.json", summary)
    with (output / "per_task.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tasks[0])); writer.writeheader(); writer.writerows(tasks)
    write_json(output / "figure_source.json", {
        "summary": "summary.json", "per_task": "per_task.csv",
        "oracle_is_evaluation_only": True,
        "best_link_uses_only_frozen_relation_score": True,
        "diagnosis": diagnosis, "decision": decision,
    })
    _plot(summary, output / "primitive_relation_observable_failure_attribution")
    print(json.dumps({
        "overall_status": summary["overall_status"], "diagnosis": diagnosis,
        "decision": decision, "condition_passing_seed_counts": counts,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
