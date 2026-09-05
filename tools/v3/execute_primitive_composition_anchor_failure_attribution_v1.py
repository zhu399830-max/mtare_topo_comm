#!/usr/bin/env python3
"""Attribute the three-seed composition-anchor failure on frozen C07 only."""

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

from mtare_topo.data.primitive_composition_anchor_batches import CompositionAnchorBatchLoader
from mtare_topo.evaluation.primitive_composition_anchor_failure_attribution import (
    CORRECTION_EDGES_M,
    DISTANCE_EDGES_M,
    SLOPE_EDGES_DEG,
    binned_error_summary,
    categorical_error_summary,
    diagnose_attribution,
    symmetric_negative_pair_distance,
)
from mtare_topo.evaluation.primitive_relation_sparse_port_failure_attribution import (
    exact_ranked_selection,
    score_quantiles,
)
from mtare_topo.representation.primitive_composition_anchor_model import (
    FrozenObservableCompositionAnchorNet,
    composition_anchor_safe_score,
    gaussian_anchor_compatibility_logits,
)
from mtare_topo.representation.primitive_composition_anchor_training import (
    align_composition_anchor_targets,
    composition_anchor_numpy_batch_to_torch,
)
from mtare_topo.representation.primitive_relation_losses import (
    align_primitive_relation_targets,
    match_primitives,
)
from mtare_topo.representation.primitive_relation_observable_model import (
    ObservableSparsePortRelationNet,
)


EXPECTED_ROWS = 64_644
EXPECTED_TASKS = 30
EXPECTED_POSITIVES = 442_936
SEEDS = (0, 1, 2)
BATCH_SIZE = 128
SAFE_PRECISION = 0.98


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_model(path: Path, seed: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != "primitive_composition_anchor_checkpoint_v1"
        or int(checkpoint.get("seed", -1)) != seed
        or checkpoint.get("training_contract", {}).get("anchor_target_frame")
        != "current_sensor_xyz_yaw_only"
        or checkpoint.get("training_contract", {}).get("frozen_observable_backbone") is not True
    ):
        raise RuntimeError("composition-anchor checkpoint identity drift")
    model = FrozenObservableCompositionAnchorNet(ObservableSparsePortRelationNet())
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def _cross_primitive_upper(device: torch.device) -> torch.Tensor:
    endpoint = torch.arange(64, device=device)
    primitive = endpoint // 2
    return (
        (primitive[:, None] != primitive[None, :])
        & torch.triu(torch.ones(64, 64, dtype=torch.bool, device=device), diagonal=1)
    )


class _ConditionBuffer:
    def __init__(self) -> None:
        self.score: list[np.ndarray] = []
        self.target: list[np.ndarray] = []
        self.overlap: list[np.ndarray] = []

    def append(self, score: torch.Tensor, target: torch.Tensor, overlap: torch.Tensor) -> None:
        self.score.append(score.detach().cpu().numpy().astype(np.float32, copy=False))
        self.target.append(target.detach().cpu().numpy().astype(np.bool_, copy=False))
        self.overlap.append(overlap.detach().cpu().numpy().astype(np.bool_, copy=False))

    def finalize(self) -> dict:
        score = np.concatenate(self.score)
        target = np.concatenate(self.target)
        overlap = np.concatenate(self.overlap)
        best = exact_ranked_selection(score, target, total_positive=EXPECTED_POSITIVES)
        safe = exact_ranked_selection(
            score, target, total_positive=EXPECTED_POSITIVES,
            minimum_precision=SAFE_PRECISION,
        )

        def overlap_fp(selection: dict) -> int:
            if not selection["available"]:
                return 0
            return int(np.count_nonzero(overlap & ~target & (score >= selection["threshold"])))

        return {
            "candidate_pairs": int(len(target)),
            "candidate_true_pairs": int(np.count_nonzero(target)),
            "best_f1": best,
            "safe": safe,
            "score_quantiles": score_quantiles(score, target),
            "overlap_hard_negative_fp_at_best_f1": overlap_fp(best),
            "overlap_hard_negative_fp_at_safe": overlap_fp(safe),
        }


class _EndpointStrata:
    def __init__(self) -> None:
        self.raw: list[np.ndarray] = []
        self.anchor: list[np.ndarray] = []
        self.distance: list[np.ndarray] = []
        self.correction: list[np.ndarray] = []
        self.slope: list[np.ndarray] = []
        self.cluster_size: list[np.ndarray] = []
        self.family: list[np.ndarray] = []
        self.scale: list[np.ndarray] = []

    def append(
        self, *, raw, anchor, distance, correction, slope, cluster_size, family, scale,
    ) -> None:
        for name, value in locals().copy().items():
            if name not in {"self", "family"}:
                getattr(self, name).append(value.detach().cpu().numpy().astype(np.float64, copy=False))
        self.family.append(np.full(len(raw), family, dtype=np.int8))

    @staticmethod
    def _summary(value: np.ndarray) -> dict:
        return {
            "count": int(len(value)), "mean": float(np.mean(value)),
            "median": float(np.median(value)), "p90": float(np.quantile(value, 0.9)),
            "p99": float(np.quantile(value, 0.99)), "maximum": float(np.max(value)),
        }

    def finalize(self) -> dict:
        values = {
            name: np.concatenate(getattr(self, name))
            for name in ("raw", "anchor", "distance", "correction", "slope", "cluster_size", "family", "scale")
        }
        raw, anchor = values["raw"], values["anchor"]
        raw_mean, anchor_mean = float(np.mean(raw)), float(np.mean(anchor))
        return {
            "observed_endpoints": int(len(raw)),
            "raw_endpoint_error_m": self._summary(raw),
            "predicted_anchor_error_m": self._summary(anchor),
            "predicted_scale_m": self._summary(values["scale"]),
            "relative_mean_error_improvement": (raw_mean - anchor_mean) / raw_mean,
            "by_teacher_anchor_distance_m": binned_error_summary(
                raw, anchor, values["distance"], DISTANCE_EDGES_M,
            ),
            "by_required_correction_m": binned_error_summary(
                raw, anchor, values["correction"], CORRECTION_EDGES_M,
            ),
            "by_absolute_axis_slope_deg": binned_error_summary(
                raw, anchor, values["slope"], SLOPE_EDGES_DEG,
            ),
            "by_observable_attachment_cluster_size": categorical_error_summary(
                raw, anchor, values["cluster_size"],
                {index: str(index) for index in sorted(set(values["cluster_size"].astype(int)))},
            ),
            "by_geometry_family": categorical_error_summary(
                raw, anchor, values["family"],
                {0: "ellipse", 1: "rounded_rectangle", 2: "c1_mixed"},
            ),
        }


def _family(task_name: str) -> int:
    lower = task_name.lower()
    if "rounded" in lower or "rectangle" in lower:
        return 1
    if "mixed" in lower or "c1" in lower:
        return 2
    return 0


@torch.no_grad()
def _evaluate_seed(seed, model, loader, existence_threshold, device):
    names = (
        "deployed_model_safe",
        "proposal_oracle_model_safe",
        "proposal_oracle_model_compatibility",
        "proposal_oracle_predicted_anchor_distance",
        "proposal_oracle_raw_endpoint_distance",
        "proposal_oracle_teacher_anchor_gaussian",
        "proposal_oracle_teacher_anchor_safe",
        "proposal_oracle_teacher_anchor_distance",
    )
    buffers = {name: _ConditionBuffer() for name in names}
    strata = _EndpointStrata()
    rows = positive_count = 0
    upper = _cross_primitive_upper(device)
    per_task = []
    peak_cuda = 0
    for task_name in loader.task_names:
        task_rows = loader._lengths[task_name]
        task_raw, task_anchor = [], []
        for start in range(0, task_rows, BATCH_SIZE):
            indices = np.arange(start, min(start + BATCH_SIZE, task_rows), dtype=np.int64)
            numpy_batch = loader._read(task_name, indices)
            batch = composition_anchor_numpy_batch_to_torch(numpy_batch, device=device)
            prediction = model(
                batch.base.range_valid,
                batch.base.relative_translation_current_sensor_m,
                batch.base.relative_yaw_current_sensor_deg,
            )
            assignments = match_primitives(prediction.primitive, batch.base.targets)
            aligned = align_composition_anchor_targets(
                batch.anchor_current_sensor_m, batch.base.targets, assignments,
            )
            relation = align_primitive_relation_targets(batch.base.targets, assignments)
            observed = aligned.endpoint_observed.bool()
            observed_flat = observed.reshape(len(observed), 64)
            validity = observed_flat[:, :, None] & observed_flat[:, None, :] & upper[None]
            target = aligned.attachment.reshape(len(observed), 64, 64).bool() & validity
            overlap = aligned.disconnected_overlap.bool().repeat_interleave(2, 1).repeat_interleave(2, 2)
            positive_count += int(torch.count_nonzero(target))

            raw_point = prediction.primitive.axis_control_current_sensor_m[:, :, (0, 2)]
            predicted_anchor = prediction.composition.anchor_current_sensor_m
            teacher_anchor = aligned.anchor_current_sensor_m
            scale = prediction.composition.scale_m
            raw_error = torch.linalg.vector_norm(raw_point[observed] - teacher_anchor[observed], dim=-1)
            anchor_error = torch.linalg.vector_norm(predicted_anchor[observed] - teacher_anchor[observed], dim=-1)
            task_raw.append(raw_error.cpu().numpy()); task_anchor.append(anchor_error.cpu().numpy())

            axis = relation["axis"]
            tangent = axis[:, :, 2] - axis[:, :, 0]
            horizontal = torch.linalg.vector_norm(tangent[..., :2], dim=-1)
            slope = torch.rad2deg(torch.atan2(torch.abs(tangent[..., 2]), horizontal)).unsqueeze(-1).expand(-1, -1, 2)
            cluster_size = target.sum(dim=-1).reshape(len(observed), 32, 2)
            strata.append(
                raw=raw_error,
                anchor=anchor_error,
                distance=torch.linalg.vector_norm(teacher_anchor[observed], dim=-1),
                correction=torch.linalg.vector_norm((teacher_anchor - raw_point)[observed], dim=-1),
                slope=slope[observed],
                cluster_size=cluster_size[observed].to(torch.float32),
                family=_family(task_name),
                scale=scale[observed],
            )

            model_safe = composition_anchor_safe_score(prediction)
            model_compatibility = torch.sigmoid(prediction.composition.compatibility_logits)
            evidence = torch.sigmoid(prediction.primitive.endpoint_evidence_logits).reshape(len(observed), 64)
            evidence_pair = evidence[:, :, None] * evidence[:, None, :]
            predicted_anchor_distance = symmetric_negative_pair_distance(predicted_anchor.reshape(len(observed), 64, 3))
            raw_endpoint_distance = symmetric_negative_pair_distance(raw_point.reshape(len(observed), 64, 3))
            teacher_anchor_distance = symmetric_negative_pair_distance(teacher_anchor.reshape(len(observed), 64, 3))
            teacher_logits = gaussian_anchor_compatibility_logits(
                teacher_anchor, scale,
                log_temperature=model.anchor_head.compatibility_log_temperature,
                bias=model.anchor_head.compatibility_bias,
            )
            teacher_gaussian = torch.sigmoid(teacher_logits)
            teacher_safe = teacher_gaussian * evidence_pair
            active = torch.sigmoid(prediction.primitive.existence_logits) >= existence_threshold
            deployed_endpoint = active.repeat_interleave(2, 1)
            deployed_eligible = validity & deployed_endpoint[:, :, None] & deployed_endpoint[:, None, :]
            conditions = {
                "deployed_model_safe": (model_safe, deployed_eligible),
                "proposal_oracle_model_safe": (model_safe, validity),
                "proposal_oracle_model_compatibility": (model_compatibility, validity),
                "proposal_oracle_predicted_anchor_distance": (predicted_anchor_distance, validity),
                "proposal_oracle_raw_endpoint_distance": (raw_endpoint_distance, validity),
                "proposal_oracle_teacher_anchor_gaussian": (teacher_gaussian, validity),
                "proposal_oracle_teacher_anchor_safe": (teacher_safe, validity),
                "proposal_oracle_teacher_anchor_distance": (teacher_anchor_distance, validity),
            }
            for name, (score, eligible) in conditions.items():
                buffers[name].append(score[eligible], target[eligible], (overlap & ~target)[eligible])
            rows += len(indices)
            if torch.cuda.is_available():
                peak_cuda = max(peak_cuda, int(torch.cuda.max_memory_reserved(device)))
        raw_task = np.concatenate(task_raw); anchor_task = np.concatenate(task_anchor)
        per_task.append({
            "seed": seed, "task": task_name, "rows": task_rows,
            "geometry_family": ("ellipse", "rounded_rectangle", "c1_mixed")[_family(task_name)],
            "observed_endpoints": int(len(raw_task)),
            "raw_endpoint_mean_error_m": float(np.mean(raw_task)),
            "predicted_anchor_mean_error_m": float(np.mean(anchor_task)),
            "relative_improvement": float((np.mean(raw_task) - np.mean(anchor_task)) / np.mean(raw_task)),
        })
    if rows != EXPECTED_ROWS or positive_count != EXPECTED_POSITIVES:
        raise RuntimeError(f"C07 attribution population drift: {rows}/{positive_count}")
    return {
        "seed": seed, "rows": rows, "tasks": len(loader.task_names),
        "existence_threshold": existence_threshold,
        "conditions": {name: value.finalize() for name, value in buffers.items()},
        "endpoint_error_strata": strata.finalize(),
        "peak_cuda_reserved_bytes": peak_cuda,
    }, per_task


def _plot(summary: dict, output: Path) -> None:
    short = (
        ("deployed_model_safe", "full model"),
        ("proposal_oracle_model_safe", "GT proposals"),
        ("proposal_oracle_predicted_anchor_distance", "predicted anchor distance"),
        ("proposal_oracle_teacher_anchor_distance", "Teacher anchor distance"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
    x = np.arange(3); width = 0.19
    for index, (name, label) in enumerate(short):
        axes[0].bar(
            x + (index - 1.5) * width,
            [row["conditions"][name]["best_f1"]["f1"] for row in summary["seeds"]],
            width=width, label=label,
        )
    axes[0].set_xticks(x, [f"seed{i}" for i in range(3)])
    axes[0].set_ylabel("best C07 attachment F1"); axes[0].legend(fontsize=7)
    axes[1].bar(
        x - 0.17,
        [row["endpoint_error_strata"]["raw_endpoint_error_m"]["mean"] for row in summary["seeds"]],
        width=0.34, label="frozen raw endpoint",
    )
    axes[1].bar(
        x + 0.17,
        [row["endpoint_error_strata"]["predicted_anchor_error_m"]["mean"] for row in summary["seeds"]],
        width=0.34, label="learned anchor",
    )
    axes[1].set_xticks(x, [f"seed{i}" for i in range(3)])
    axes[1].set_ylabel("mean endpoint-to-Teacher error [m]"); axes[1].legend(fontsize=8)
    fig.suptitle("C07 composition-anchor failure attribution")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"primitive_composition_anchor_failure_attribution.{suffix}", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--sensor-root", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--observability-root", required=True, type=Path)
    parser.add_argument("--anchor-root", required=True, type=Path)
    parser.add_argument("--formal-evaluation-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("composition-anchor attribution requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    loader = CompositionAnchorBatchLoader(
        args.sensor_root / "c07", args.teacher_root / "c07",
        args.observability_root / "c07", args.anchor_root / "c07",
    )
    if len(loader) != EXPECTED_ROWS or len(loader.task_names) != EXPECTED_TASKS:
        raise RuntimeError("C07 attribution loader population drift")
    formal = json.loads(args.formal_evaluation_summary.read_text(encoding="utf-8"))
    if formal.get("decision") != "STOP_COMPOSITION_ANCHOR_BEFORE_C08_AND_GRAPH":
        raise RuntimeError("formal composition-anchor failure source drift")
    formal_seeds = {int(row["seed"]): row for row in formal["c07"]["seeds"]}
    seeds, per_task = [], []
    for seed in SEEDS:
        model, checkpoint = _load_model(args.models_root / f"seed{seed}/selected.pt", seed, device)
        threshold = float(formal_seeds[seed]["metrics"]["existence_threshold"])
        result, task_rows = _evaluate_seed(seed, model, loader, threshold, device)
        reproduced = result["conditions"]["deployed_model_safe"]["best_f1"]
        expected = formal_seeds[seed]["metrics"]["pair_metrics"]["deployed"]["safe"]["best_f1"]
        for key in ("true_positive", "false_positive", "false_negative", "selected_pairs"):
            if int(reproduced[key]) != int(expected[key]):
                raise RuntimeError(f"seed{seed} formal metric reproduction drift: {key}")
        if abs(float(reproduced["f1"]) - float(expected["f1"])) > 1e-12:
            raise RuntimeError(f"seed{seed} formal F1 reproduction drift")
        result["selected_epoch"] = int(checkpoint["epoch"])
        result["formal_deployed_metrics_reproduced"] = True
        seeds.append(result); per_task.extend(task_rows)
        _write(output / f"seed{seed}.json", result)
        print(json.dumps({
            "seed": seed,
            "deployed_f1": reproduced["f1"],
            "predicted_anchor_safe_tp": result["conditions"]["proposal_oracle_predicted_anchor_distance"]["safe"]["true_positive"],
            "teacher_anchor_safe_tp": result["conditions"]["proposal_oracle_teacher_anchor_distance"]["safe"]["true_positive"],
        }), flush=True)
    diagnosis = diagnose_attribution(seeds)
    summary = {
        "schema_version": "primitive_composition_anchor_failure_attribution_v1",
        "system_evidence_pass": True,
        "scientific_model_pass": False,
        "rows_per_seed": EXPECTED_ROWS,
        "model_forward_rows": EXPECTED_ROWS * 3,
        "tasks_per_seed": EXPECTED_TASKS,
        "observable_target_positives_per_seed": EXPECTED_POSITIVES,
        "seeds": seeds,
        **diagnosis,
        "formal_deployed_metrics_reproduced": all(row["formal_deployed_metrics_reproduced"] for row in seeds),
        "optimizer_steps": 0, "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        "graph_replays": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
        "peak_cuda_reserved_bytes": max(row["peak_cuda_reserved_bytes"] for row in seeds),
    }
    _write(output / "summary.json", summary)
    _write(output / "per_task.json", per_task)
    with (output / "per_task.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_task[0]))
        writer.writeheader(); writer.writerows(per_task)
    _write(output / "figure_source.json", {
        "seeds": [{
            "seed": row["seed"], "conditions": row["conditions"],
            "endpoint_error_strata": row["endpoint_error_strata"],
        } for row in seeds],
    })
    _plot(summary, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
