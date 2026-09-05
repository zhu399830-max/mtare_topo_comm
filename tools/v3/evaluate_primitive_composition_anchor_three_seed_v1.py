#!/usr/bin/env python3
"""Frozen C07 gate for three composition-anchor head checkpoints."""

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

from mtare_topo.data.primitive_composition_anchor_batches import (
    CompositionAnchorBatchLoader,
)
from mtare_topo.evaluation.primitive_relation_sparse_port_failure_attribution import (
    exact_ranked_selection,
    score_quantiles,
)
from mtare_topo.representation.primitive_composition_anchor_model import (
    FrozenObservableCompositionAnchorNet,
    composition_anchor_safe_score,
)
from mtare_topo.representation.primitive_composition_anchor_training import (
    align_composition_anchor_targets,
    composition_anchor_numpy_batch_to_torch,
)
from mtare_topo.representation.primitive_relation_losses import match_primitives
from mtare_topo.representation.primitive_relation_observable_model import (
    ObservableSparsePortRelationNet,
)


PASS = "PASS_PRIMITIVE_COMPOSITION_ANCHOR_THREE_SEED_C07_V1"
FAIL = "FAIL_PRIMITIVE_COMPOSITION_ANCHOR_THREE_SEED_C07_V1"
EXPECTED_ROWS = 64_644
EXPECTED_TASKS = 30
EXPECTED_OBSERVABLE_POSITIVES = 442_936
EVALUATION_BATCH_SIZE = 128
MINIMUM_SAFE_PRECISION = 0.98
MINIMUM_F1_GAIN = 0.05
MINIMUM_ANCHOR_MAE_IMPROVEMENT = 0.10


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_model(path: Path, *, seed: int, device: torch.device):
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
    model.to(device).eval()
    return model, checkpoint


class _PairBuffer:
    def __init__(self) -> None:
        self.safe_score = []
        self.compatibility = []
        self.target = []
        self.overlap_hard_negative = []

    def append(self, safe, compatibility, target, overlap) -> None:
        self.safe_score.append(safe.detach().cpu().numpy().astype(np.float32, copy=False))
        self.compatibility.append(compatibility.detach().cpu().numpy().astype(np.float32, copy=False))
        self.target.append(target.detach().cpu().numpy().astype(np.bool_, copy=False))
        self.overlap_hard_negative.append(overlap.detach().cpu().numpy().astype(np.bool_, copy=False))

    def finalize(self) -> dict:
        safe = np.concatenate(self.safe_score)
        compatibility = np.concatenate(self.compatibility)
        target = np.concatenate(self.target)
        overlap = np.concatenate(self.overlap_hard_negative)
        result = {}
        for name, score in (("safe", safe), ("compatibility_only", compatibility)):
            best = exact_ranked_selection(
                score, target, total_positive=EXPECTED_OBSERVABLE_POSITIVES,
            )
            selected_safe = exact_ranked_selection(
                score, target, total_positive=EXPECTED_OBSERVABLE_POSITIVES,
                minimum_precision=MINIMUM_SAFE_PRECISION,
            )
            result[name] = {
                "best_f1": best,
                "safe": selected_safe,
                "score_quantiles": score_quantiles(score, target),
                "overlap_hard_negative_fp_at_best_f1": int(np.count_nonzero(
                    overlap & ~target & (score >= best["threshold"])
                )) if best["available"] else 0,
                "overlap_hard_negative_fp_at_safe": int(np.count_nonzero(
                    overlap & ~target & (score >= selected_safe["threshold"])
                )) if selected_safe["available"] else 0,
            }
        result["candidate_pairs"] = int(len(target))
        result["candidate_true_pairs"] = int(np.count_nonzero(target))
        return result


def _cross_primitive_upper(device: torch.device) -> torch.Tensor:
    endpoint = torch.arange(64, device=device)
    primitive = endpoint // 2
    return (
        (primitive[:, None] != primitive[None, :])
        & torch.triu(torch.ones(64, 64, dtype=torch.bool, device=device), diagonal=1)
    )


@torch.no_grad()
def _evaluate_seed(model, loader, *, existence_threshold: float, device: torch.device):
    buffers = {name: _PairBuffer() for name in ("deployed", "proposal_oracle")}
    raw_errors = []
    anchor_errors = []
    predicted_scales = []
    rows = all_positive = 0
    legacy_score_max_asymmetry = 0.0
    corrected_score_max_asymmetry = 0.0
    corrected_score_exact_symmetric = True
    corrected_score_all_finite = True
    upper = _cross_primitive_upper(device)
    for numpy_batch in loader.iter_epoch(
        batch_size=EVALUATION_BATCH_SIZE, seed=0, epoch=0, shuffle=False,
    ):
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
        observed = aligned.endpoint_observed.bool()
        raw = prediction.primitive.axis_control_current_sensor_m[:, :, (0, 2)]
        raw_errors.append(torch.linalg.vector_norm(
            raw[observed] - aligned.anchor_current_sensor_m[observed], dim=-1,
        ).cpu().numpy())
        anchor_errors.append(torch.linalg.vector_norm(
            prediction.composition.anchor_current_sensor_m[observed]
            - aligned.anchor_current_sensor_m[observed], dim=-1,
        ).cpu().numpy())
        predicted_scales.append(prediction.composition.scale_m[observed].cpu().numpy())

        observed_flat = observed.reshape(len(observed), 64)
        validity = observed_flat[:, :, None] & observed_flat[:, None, :] & upper[None]
        target = aligned.attachment.reshape(len(observed), 64, 64).bool() & validity
        all_positive += int(torch.count_nonzero(target))
        overlap = (
            aligned.disconnected_overlap.bool()
            .repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        )
        safe = composition_anchor_safe_score(prediction)
        compatibility = torch.sigmoid(prediction.composition.compatibility_logits)
        evidence = torch.sigmoid(prediction.primitive.endpoint_evidence_logits).reshape(
            len(observed), 64,
        )
        legacy_score = compatibility * evidence[:, :, None] * evidence[:, None, :]
        legacy_score_max_asymmetry = max(
            legacy_score_max_asymmetry,
            float((legacy_score - legacy_score.transpose(1, 2)).abs().max()),
        )
        corrected_score_max_asymmetry = max(
            corrected_score_max_asymmetry,
            float((safe - safe.transpose(1, 2)).abs().max()),
        )
        corrected_score_exact_symmetric = corrected_score_exact_symmetric and torch.equal(
            safe, safe.transpose(1, 2),
        )
        corrected_score_all_finite = corrected_score_all_finite and bool(
            torch.isfinite(safe).all()
        )
        active_masks = {
            "deployed": torch.sigmoid(prediction.primitive.existence_logits) >= existence_threshold,
            "proposal_oracle": aligned.primitive_mask.bool(),
        }
        for name, active in active_masks.items():
            endpoint_active = active.repeat_interleave(2, dim=1)
            eligible = validity & endpoint_active[:, :, None] & endpoint_active[:, None, :]
            buffers[name].append(
                safe[eligible], compatibility[eligible], target[eligible],
                (overlap & ~target)[eligible],
            )
        rows += len(numpy_batch.base.base.range_valid)
    if rows != EXPECTED_ROWS or all_positive != EXPECTED_OBSERVABLE_POSITIVES:
        raise RuntimeError(f"composition-anchor C07 population drift: {rows}/{all_positive}")
    raw_error = np.concatenate(raw_errors).astype(np.float64)
    anchor_error = np.concatenate(anchor_errors).astype(np.float64)
    scale = np.concatenate(predicted_scales).astype(np.float64)

    def summarize(value):
        return {
            "count": int(len(value)), "mean_m": float(np.mean(value)),
            "median_m": float(np.median(value)), "p90_m": float(np.quantile(value, .9)),
            "p99_m": float(np.quantile(value, .99)), "maximum_m": float(np.max(value)),
        }

    raw_summary = summarize(raw_error)
    anchor_summary = summarize(anchor_error)
    return {
        "rows": rows,
        "existence_threshold": existence_threshold,
        "raw_predicted_endpoint_error": raw_summary,
        "composition_anchor_error": anchor_summary,
        "anchor_mae_relative_improvement": (
            raw_summary["mean_m"] - anchor_summary["mean_m"]
        ) / raw_summary["mean_m"],
        "predicted_scale": summarize(scale),
        "safe_score_numerical_contract": {
            "legacy_left_associated_max_asymmetry": legacy_score_max_asymmetry,
            "corrected_max_asymmetry": corrected_score_max_asymmetry,
            "corrected_exact_symmetric": corrected_score_exact_symmetric,
            "corrected_all_finite": corrected_score_all_finite,
        },
        "pair_metrics": {name: value.finalize() for name, value in buffers.items()},
    }


def _plot(summary, output: Path) -> None:
    seeds = summary["c07"]["seeds"]
    baseline = summary["c07"]["baseline_attachment_f1"]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1))
    labels = ["baseline", *[f"seed{row['seed']}" for row in seeds]]
    f1 = [baseline, *[
        row["metrics"]["pair_metrics"]["deployed"]["safe"]["best_f1"]["f1"]
        for row in seeds
    ]]
    axes[0].bar(labels, f1, color=("#777777", "#287271", "#d37524", "#5b5f97"))
    axes[0].axhline(baseline + MINIMUM_F1_GAIN, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("C07 attachment F1")
    safe_tp = [
        row["metrics"]["pair_metrics"]["deployed"]["safe"]["safe"]["true_positive"]
        for row in seeds
    ]
    axes[1].bar(labels[1:], safe_tp, color=("#287271", "#d37524", "#5b5f97"))
    axes[1].set_ylabel("true links at precision ≥ 0.98")
    x = np.arange(3); width = .34
    raw = [row["metrics"]["raw_predicted_endpoint_error"]["mean_m"] for row in seeds]
    corrected = [row["metrics"]["composition_anchor_error"]["mean_m"] for row in seeds]
    axes[2].bar(x - width / 2, raw, width, label="frozen raw endpoint")
    axes[2].bar(x + width / 2, corrected, width, label="learned anchor")
    axes[2].set_xticks(x, [f"seed{i}" for i in range(3)])
    axes[2].set_ylabel("observed endpoint anchor MAE [m]")
    axes[2].legend(fontsize=8)
    fig.suptitle("Composition-anchor C07 gate (unseen topology, frozen backbone)")
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
    parser.add_argument("--anchor-root", required=True, type=Path)
    parser.add_argument("--baseline-c07-summary", required=True, type=Path)
    parser.add_argument("--existence-threshold-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("composition-anchor C07 evaluation requires CUDA")
    device = torch.device("cuda")
    loader = CompositionAnchorBatchLoader(
        args.sensor_root / "c07", args.teacher_root / "c07",
        args.observability_root / "c07", args.anchor_root / "c07",
    )
    if len(loader) != EXPECTED_ROWS or len(loader.task_names) != EXPECTED_TASKS:
        raise RuntimeError("composition-anchor evaluator C07 loader drift")
    baseline = json.loads(args.baseline_c07_summary.read_text(encoding="utf-8"))
    if (
        baseline.get("rows") != EXPECTED_ROWS
        or baseline.get("scientific_pass") is not True
        or baseline.get("observable_attachment_target_positives") != EXPECTED_OBSERVABLE_POSITIVES
    ):
        raise RuntimeError("composition-anchor non-learning baseline drift")
    baseline_f1 = float(baseline["attachment"]["f1"])
    threshold_source = json.loads(args.existence_threshold_source.read_text(encoding="utf-8"))
    thresholds = {int(row["seed"]): float(row["existence_threshold"]) for row in threshold_source["seeds"]}
    if set(thresholds) != {0, 1, 2}:
        raise RuntimeError("composition-anchor existence thresholds drift")
    seeds = []
    for seed in range(3):
        model, checkpoint = _load_model(
            args.models_root / f"seed{seed}/selected.pt", seed=seed, device=device,
        )
        metrics = _evaluate_seed(
            model, loader, existence_threshold=thresholds[seed], device=device,
        )
        deployed = metrics["pair_metrics"]["deployed"]["safe"]
        checks = {
            "attachment_f1_gain_ge_0p05": deployed["best_f1"]["f1"] - baseline_f1 >= MINIMUM_F1_GAIN,
            "safe_precision_ge_0p98_nonzero": deployed["safe"]["available"]
                and deployed["safe"]["precision"] >= MINIMUM_SAFE_PRECISION
                and deployed["safe"]["true_positive"] > 0,
            "anchor_mae_improvement_ge_10pct": metrics["anchor_mae_relative_improvement"]
                >= MINIMUM_ANCHOR_MAE_IMPROVEMENT,
            "frozen_backbone_contract": checkpoint["training_contract"]["frozen_observable_backbone"] is True,
        }
        record = {
            "seed": seed, "selected_epoch": int(checkpoint["epoch"]),
            "metrics": metrics, "checks": checks, "pass": all(checks.values()),
        }
        seeds.append(record)
        _write(output / f"c07_seed{seed}.json", record)
        print(json.dumps({"seed": seed, "checks": checks, "deployed": deployed}), flush=True)
    passing = sum(row["pass"] for row in seeds)
    scientific_pass = passing >= 2
    summary = {
        "schema_version": "primitive_composition_anchor_three_seed_evaluation_v1",
        "overall_status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "c07": {
            "baseline_attachment_f1": baseline_f1,
            "seeds": seeds,
            "passing_seeds": passing,
            "scientific_pass": scientific_pass,
        },
        "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        "graph_replays": 0, "mtare_worlds_read": 0,
        "decision": (
            "ALLOW_C08_COMPOSITION_ANCHOR_ZERO_ADAPTATION_TRANSFER"
            if scientific_pass else "STOP_COMPOSITION_ANCHOR_BEFORE_C08_AND_GRAPH"
        ),
        "duration_seconds": time.monotonic() - started,
    }
    _write(output / "summary.json", summary)
    _plot(summary, output / "primitive_composition_anchor_c07_comparison")
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
